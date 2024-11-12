##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

# from fs_plugins.models.fairseq_nat_model_mod import FairseqNATModel
from fairseq.models.nat.fairseq_nat_model import FairseqNATModel

import logging
import random
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn, jit
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import (
    PositionalEmbedding,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from contextlib import contextmanager

logger = logging.getLogger(__name__)

import nltk
# from numba import njit
# import numba

from rouge_score import rouge_scorer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        

def load_libnat():
    try:
        from fairseq import libnat_cuda

        return libnat_cuda, True

    except ImportError as e:
        print(str(e) + "... fall back to CPU version")

        try:
            from fairseq import libnat

            return libnat, False

        except ImportError as e:
            import sys

            sys.stderr.write(
                "ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n"
            )
            raise e


# @njit(nopython=True, parallel=True)
def reranker_bleu(encoder_out, G2_beams, model=None):
    
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    smoothing_function=chencherry.method1

    bsz, beam_size_ext = G2_beams.shape[:2]

    reranking_score = np.zeros((bsz, beam_size_ext))
    for i in range(bsz):
        for j in range(beam_size_ext):
            if model is not None:
                src_tokens = [model.id2word[int(x)]  for x in encoder_out[i] if x not in [0, 1, 2]]
                tgt_tokens = [model.id2word[int(x)]  for x in G2_beams[i][j] if x not in [0, 1, 2]]
            else:
                src_tokens = [int(x)  for x in encoder_out[i] if x not in [0, 1, 2]]
                tgt_tokens = [int(x)  for x in G2_beams[i][j] if x not in [0, 1, 2]]
            reranking_score[i,j] = nltk.translate.bleu_score.sentence_bleu([src_tokens], tgt_tokens, smoothing_function=smoothing_function, weights=(0.25, 0.25, 0.25, 0.25))
    
    return reranking_score

def reranker_rouge(rouge_model, encoder_out, G2_beams):
    bsz, beam_size_ext = G2_beams.shape[:2]

    reranking_score = np.zeros((bsz, beam_size_ext))
    for i in range(bsz):
        for j in range(beam_size_ext):
            src_tokens = ' '.join([str(x)  for x in encoder_out[i] if x not in [0, 1, 2]])
            tgt_tokens = ' '.join([str(x)  for x in G2_beams[i][j] if x not in [0, 1, 2]])
            reranking_score[i,j] = rouge_model.score(tgt_tokens, src_tokens)['rouge1'].fmeasure
    return reranking_score


def reranker_ngram(G2_beams, id2word, ngram_model):
    bsz, beam_size_ext = G2_beams.shape[:2]

    reranking_score = np.zeros((bsz, beam_size_ext))
    for i in range(bsz):
        for j in range(beam_size_ext):
            tgt_tokens = ' '.join([id2word[x] for x in G2_beams[i][j] if x not in [0, 1, 2]])
            reranking_score[i,j] = ngram_model.score(tgt_tokens)
    
    return reranking_score

def reranker_lev(lev_model, src_tokens, tgt_tokens):
    bsz, beam_size_ext, beam_len = tgt_tokens.shape

    # reranking_score = np.zeros((bsz, beam_size_ext))
    
    src_tokens_expand = src_tokens.repeat_interleave(beam_size_ext, dim=0)
    src_tokens_expand_len = (src_tokens_expand != 2).sum(-1)
    
    tgt_tokens = tgt_tokens.reshape(-1, beam_len)
    tgt_tokens_len = (tgt_tokens != 2).sum(-1)
    edit_dis = lev_model.levenshtein_distance(
                    src_tokens_expand.int(),
                    tgt_tokens.int(),
                    src_tokens_expand_len.int(),
                    tgt_tokens_len.int()
                )
    edit_dis[edit_dis==3] = 0
    edit_dis[edit_dis!=0] = 1
    edit_dis = edit_dis.sum(-1)
    edit_dis = 1 - edit_dis / (src_tokens_expand_len + tgt_tokens_len)
    return edit_dis.reshape(bsz, beam_size_ext) 

def reranker_bert(encoder_out, G2_beams, model, device):
    bsz, beam_size_ext = G2_beams.shape[:2]
    concat_sample_list = []
    for i in range(bsz):
        src_tokens = ' '.join([model.id2word[int(x)]  for x in encoder_out if x not in [1]])
        for j in range(beam_size_ext):
            tgt_tokens = ' '.join([model.id2word[int(x)]  for x in G2_beams[i][j] if x not in [1]])
            concat_sample_list.append(src_tokens + ' ' + tgt_tokens + ' </s>')

    reranker_self_attention = getattr(model.cfg, "reranker_self_attention", False)
    reranker_self_attention_linear = getattr(model.cfg, "reranker_self_attention_linear", False)
    reranker_self_attention_pos_emb = getattr(model.cfg, "reranker_self_attention_pos_emb", False)
    
    reranker_input = model.reranker_tokenizer(concat_sample_list, padding=True, return_tensors='pt').to(device)
    
    if not reranker_self_attention:
        pred_rank = model.reranker_scorer(model.reranker_model(**reranker_input).pooler_output).squeeze(1).log_softmax(-1)
    else:
        reranker_feature = model.reranker_model(**reranker_input).pooler_output
        
        if reranker_self_attention_pos_emb:
            reranker_feature = model.reranker_self_attn_pos_emb(reranker_feature.unsqueeze(0)).squeeze(0)
        
        if reranker_self_attention_linear:
            fea_atten_out = model.reranker_self_attn(
                model.reranker_self_attn_q(reranker_feature), 
                model.reranker_self_attn_k(reranker_feature), 
                model.reranker_self_attn_v(reranker_feature))[0].mean(0)
        else:
            fea_atten_out = model.reranker_self_attn(reranker_feature, reranker_feature, reranker_feature)[0].mean(0)
        
        pred_rank = model.reranker_scorer(fea_atten_out).log_softmax(-1)

    # rank_score = torch.softmax(model.reranker_scorer(model.reranker_model(**reranker_input).pooler_output).squeeze(0), dim=0).reshape(bsz, beam_size_ext)
    # rank_score = torch.log_softmax(model.reranker_scorer(model.reranker_model(**reranker_input).pooler_output).squeeze(1), dim=0).reshape(bsz, beam_size_ext)
    rank_score = pred_rank.reshape(bsz, beam_size_ext)

    return rank_score

def reranker_bert_v2(encoder_out, G2_beams, model, device):
    bsz, beam_size_ext = G2_beams.shape[:2]
    concat_sample_list = []
    all_src_list = []
    all_tgt_list = []
    for i in range(bsz):
        src_tokens = ' '.join([model.id2word[int(x)]  for x in encoder_out if x not in [1]])
        all_src_list.append(src_tokens)
        all_tgt_list.append(''.join( ' '.join([' '.join(['<s>'] + [model.id2word[int(x)] for x in beam if int(x) not in [0, 1, 2]] + ['</s>']) for  beam in G2_beams[i]])))

    src_input = model.reranker_tokenizer(all_src_list, padding=True, return_tensors='pt', truncation=True).to(device)
    tgt_input = model.reranker_tokenizer(all_tgt_list, padding=True, return_tensors='pt', truncation=True).to(device)
    
    
    src_rep = model.reranker_model(**src_input).last_hidden_state.permute(1, 0, 2)
    
    tgt_rep = model.reranker_model(**tgt_input).last_hidden_state.permute(1, 0, 2)
    
    key_padding_mask = (1 - src_input['attention_mask']).bool()
            
    fea_atten_out = model.reranker_self_attn(
        model.reranker_self_attn_q(tgt_rep), 
        model.reranker_self_attn_k(src_rep), 
        model.reranker_self_attn_v(src_rep),
        key_padding_mask=key_padding_mask
        )[0]
    
    target_mask = tgt_input['attention_mask'] 
    target_mask = target_mask.permute(1, 0).unsqueeze(2).expand_as(fea_atten_out)
    fea_atten_out = (fea_atten_out * target_mask)
    
    fea_atten_out = fea_atten_out.mean(0)
    pred_rank = model.reranker_scorer(fea_atten_out).log_softmax(-1)

    # rank_score = torch.softmax(model.reranker_scorer(model.reranker_model(**reranker_input).pooler_output).squeeze(0), dim=0).reshape(bsz, beam_size_ext)
    # rank_score = torch.log_softmax(model.reranker_scorer(model.reranker_model(**reranker_input).pooler_output).squeeze(1), dim=0).reshape(bsz, beam_size_ext)
    rank_score = pred_rank.reshape(bsz, beam_size_ext)

    return rank_score

def reranker_bert_v3(encoder_out, G2_beams, model, device):
    def repeat_interleave(input):
            return torch.repeat_interleave(torch.arange(input.size(0), device=device), input)

    def stack_pad_2d(input):
        max_len = max([x.size(0) for x in input])
        pad_input = [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in input]
        return torch.stack(pad_input)     
    
    def stack_pad_1d(input):
        max_len = max([x.size(0) for x in input])
        pad_input = [F.pad(x, (0, max_len - x.size(0))) for x in input]
        return torch.stack(pad_input)
                
    def make_mask(lengths):
        lengths = torch.tensor(lengths, device=device)
        return torch.arange(lengths.max(), device=device)[None, :] < lengths[:, None]
        
    bsz, beam_size_ext = G2_beams.shape[:2]
    all_src_list = []
    all_tgt_list = []
    for i in range(bsz):
        src_tokens = ' '.join([model.id2word[int(x)]  for x in encoder_out[i] if x not in [1]])
        all_src_list.append(src_tokens)
        all_tgt_list.append([' '.join(['<s>'] + [model.id2word[int(x)] for x in beam if int(x) not in [0, 1, 2]] + ['</s>']) for  beam in G2_beams[i]])

    src_input = model.reranker_tokenizer(all_src_list, padding=True, return_tensors='pt', truncation=True).to(device)
    src_mask = 1 - src_input['attention_mask'].float()
    src_rep = model.reranker_model(**src_input).last_hidden_state
    
    all_beam_len_list = []
    all_beam_rep_list = []
    all_beam_sep_len_list = []
    all_beam_pos_emb_input_list = []
    for one_beam in all_tgt_list:
        one_beam_input = model.reranker_tokenizer(one_beam, padding=True, return_tensors='pt', truncation=True).to(device)
        one_beam_rep = model.reranker_model(**one_beam_input).last_hidden_state

        one_beam_lens = one_beam_input['attention_mask'].sum(-1)
        
        cat_beam_rep = torch.cat([x[:x_len] for x_len, x in zip(one_beam_lens, one_beam_rep)], dim=0)
        all_beam_rep_list.append(cat_beam_rep)
        
        all_beam_pos_emb_input_list.append(repeat_interleave(one_beam_lens))
        
        all_beam_sep_len_list.append(one_beam_lens)
        all_beam_len_list.append(one_beam_lens.sum(-1))    
            
    all_beam_rep = stack_pad_2d(all_beam_rep_list)
    all_beam_pos_emb_input = stack_pad_1d(all_beam_pos_emb_input_list)
    all_beam_mask = make_mask(all_beam_len_list)
    all_beam_mask = 1 - all_beam_mask.float()
    
    reranker_self_attention_pos_emb = getattr(model.cfg, "reranker_self_attention_pos_emb", False)
    if reranker_self_attention_pos_emb:
        all_beam_rep += model.reranker_pos_emb(all_beam_pos_emb_input)
    
    fea_atten_out = model.reranker_decoder(
            x = all_beam_rep.permute(1, 0, 2),
            encoder_out = src_rep.permute(1, 0, 2),
            encoder_padding_mask = src_mask.to(one_beam_rep.dtype),
            self_attn_padding_mask = all_beam_mask.to(one_beam_rep.dtype)
        )[0]
    
    reranker_scorer_method = getattr(model.cfg, 'reranker_scorer_method', 'classification')
    if reranker_scorer_method == 'classification':
        pred_rank = model.reranker_scorer(fea_atten_out.mean(0)).log_softmax(-1)
    elif reranker_scorer_method == 'energy':
        feat_each_sent_list = []
        for _idx, x in enumerate(fea_atten_out.permute(1, 0, 2)):
            len_pos_each_sent = torch.cumsum(all_beam_sep_len_list[_idx],dim=0)
            len_pos_each_sent = torch.cat([torch.tensor([0], device=device), len_pos_each_sent])
            feat_each_sent = [x[len_pos_each_sent[i]:len_pos_each_sent[i+1]].mean(0) for i in range(len(len_pos_each_sent)-1)]                
            feat_each_sent_list.append(torch.stack(feat_each_sent, dim=0))
        
        feat_each_sent_list = torch.stack(feat_each_sent_list, dim=0)
        pred_rank = model.reranker_scorer(feat_each_sent_list).squeeze(2).log_softmax(-1)
        
    rank_score = pred_rank.reshape(bsz, beam_size_ext)
    return rank_score

def log_bmm(a: Tensor, b: Tensor) -> Tensor:
    """Performs a batch matrix-matrix product of matrices in log-space.
    Args:
        a: tensor with shape (b, n, m)
        b: tensor with shape (b, m, p)
    Returns:
        tensor with shape (b, n, p)
    """

    assert a.ndim == b.ndim == 3
    assert a.size(0) == b.size(0)
    assert a.size(2) == b.size(1)

    bsz, p, m = a.size()
    _, _, n = b.size()
    a = a.unsqueeze(2).expand(bsz, p, n, m)
    b = b.unsqueeze(1).transpose(2, 3).expand(bsz, p, n, m)
    return torch.logsumexp(a + b, dim=-1)

class Beams:
    def __init__(self, bsz, L, K, device):
        self.beams = None # bsz * beam * []
        self.bsz = bsz
        self.L = L
        self.K = K
        self.device = device
        self.score = - torch.ones(bsz, L, K, device=self.device)  * torch.tensor(float("inf")) # bsz * L * K 
    
    def init_score(self, scores, j, path=None):
        K_vocab = scores.size(1)
        self.score[:, j, :K_vocab] = scores
        if path is not None:
            self.beams = 3 * torch.ones(self.bsz, self.K, 1, device=path.device, dtype=path.dtype)
            self.beams[:, :K_vocab] = path.unsqueeze(2)
            
            
@contextmanager
def torch_seed(seed):
    # modified from lunanlp
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("glat_decomposed_link")
class GlatDecomposedLink(FairseqNATModel):

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        GlatLinkDecoder.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

        parser.add_argument('--links-feature', type=str, default="feature:position", help="Features used to predict transition.")
        parser.add_argument('--max-transition-length', type=int, default=99999, help="Max transition distance. -1 means no limitation, \
                        which cannot be used for cuda custom operations. To use cuda operations with no limitation, please use a very large number such as 99999.")

        parser.add_argument("--src-upsample-scale", type=float, default=None, help="Specify the graph size with a upsample factor (lambda).  Graph Size = \\lambda * src_length")
        parser.add_argument("--src-upsample-fixed", type=int, default=None, help="Specify the graph size by a constant. Cannot use together with src-upsample-scale")
        parser.add_argument("--length-multiplier", type=float, default=None, help="Deprecated") # does not work now
        parser.add_argument('--max-decoder-batch-tokens', type=int, default=None, help="Max tokens for LightSeq Decoder when using --src-upsample-fixed")

        parser.add_argument('--filter-max-length', default=None, type=str, help='Filter the sample that above the max lengths, e.g., "128:256" indicating 128 for source, 256 for target. Default: None, for filtering according max-source-positions and max-target-positions')
        parser.add_argument("--filter-ratio", type=float, default=None, help="Deprecated") # does not work now; need support of trainer.py

        parser.add_argument('--decode-strategy', type=str, default="lookahead", help='One of "greedy", "lookahead", "viterbi", "jointviterbi", "beamsearch"')

        parser.add_argument('--decode-alpha', type=float, default=1.1, help="Used for length penalty. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beta', type=float, default=1, help="Scale the score of logits. log P(Y, A) := sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})")
        parser.add_argument('--decode-viterbibeta', type=float, default=1, help="Length penalty for Viterbi decoding. Viterbi decoding finds the sentence maximize: P(A,Y|X) / |Y|^{beta}")
        parser.add_argument('--decode-top-cand-n', type=float, default=5, help="Numbers of top candidates when considering transition")
        parser.add_argument('--decode-gamma', type=float, default=0.1, help="Used for n-gram language model score. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beamsize', type=float, default=100, help="Beam size")
        parser.add_argument('--decode-max-beam-per-length', type=float, default=10, help="Limits the number of beam that has a same length in each step")
        parser.add_argument('--decode-top-p', type=float, default=0.9, help="Max probability of top candidates when considering transition")
        parser.add_argument('--decode-lm-path', type=str, default=None, help="Path to n-gram language model. None for not using n-gram LM")
        parser.add_argument('--decode-max-batchsize', type=int, default=32, help="Should not be smaller than the real batch size (the value is used for memory allocation)")
        parser.add_argument('--decode-dedup', type=bool, default=False, help="Use token deduplication in BeamSearch")
        
        # for length_control
        parser.add_argument('--specified-length-fixed', type=int, default=None)
        parser.add_argument('--specified-length-ratio', type=float, default=None)
        parser.add_argument('--not-force-length', type=bool, default=False)
        
        parser.add_argument('--minimal-target-length', type=int, default=2)
        parser.add_argument('--length-beam-K', type=int, default=1)
        parser.add_argument('--length-beam-KV', type=int, default=1)
        parser.add_argument('--rerank-lambda', type=float, default=0.01)
        parser.add_argument('--rerank-method', type=str, default="none") # , choices=["bleu", 'ngram', 'rouge', 'none', 'oracle', 'lev']

        # parser.add_argument('--char-control-method', type=int, default=None, choices=["greedy", "bucket"])
        # parser.add_argument('--char-control-limit', type=int, default=None)
        # parser.add_argument('--char-control-bucket-size', type=int, default=5)
        
        parser.add_argument('--use-reranker', action='store_true', default=False)
        parser.add_argument('--reranker-pretrain-model', type=str, default="roberta") 
        parser.add_argument('--reranker-pretrain-model-random', action='store_true', default=False) 
        parser.add_argument('--reranker-inject-rank', action='store_true', default=False) 
        parser.add_argument('--reranker-self-attention', action='store_true', default=False) 
        parser.add_argument('--reranker-self-attention-linear', action='store_true', default=False)
        parser.add_argument('--reranker-self-attention-pos-emb', action='store_true', default=False)
        parser.add_argument("--reranker-scorer-method", type=str, default='classification') # classification or energy
        parser.add_argument("--reranker-progressive-method", type=str, default='none')
        
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)        
       
        self.init_beam_search()
        
        if getattr(self.args, "use_reranker", False):
            self.init_reranker()

    def init_beam_search(self):
        # if self.args.decode_strategy == "beamsearch":
        import dag_search
        self.dag_search = dag_search
        dag_search.beam_search_init(self.args.decode_max_batchsize, self.args.decode_beamsize,
                self.args.decode_top_cand_n, self.decoder.max_positions(), self.tgt_dict, self.args.decode_lm_path)
        
        word_indecies = self.tgt_dict.indices
        id2word = {}
        id2len = {}
        total_id = 0

        for word, _id in word_indecies.items():
            assert int(_id) == total_id
            id2word[total_id] = word
            if total_id > 5:
                id2len[total_id] = len(word.strip())
            else:
                id2len[total_id] = 0
            total_id += 1
        
        self.id2len = id2len
        bucket_size = getattr(self.args, "char_control_bucket_size", 5)
        self.length_bucket = [(i * bucket_size + 1, (i+1) * bucket_size) for i in range(30)]
        self.id2word = id2word
        
        rerank_method = getattr(self.args, "rerank_method", "none") 
        if rerank_method != "none":
            import kenlm
            self.ngram_model = kenlm.Model('gigaref_summary.arpa')
            self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
            self.libnat = load_libnat()[0]
    
    
    def init_reranker(self):
        reranker_pretrain_model = getattr(self.args, 'reranker_pretrain_model', 'roberta')
        if reranker_pretrain_model == 'roberta':
            from transformers import RobertaModel, RobertaTokenizer
            self.reranker_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            
            reranker_pretrain_model_random = getattr(self.args, 'reranker_pretrain_model_random', False)
            if reranker_pretrain_model_random:
                from transformers import RobertaConfig, RobertaModel
                config = RobertaConfig.from_pretrained('roberta-base')
                self.reranker_model = RobertaModel(config)
            else:            
                try:
                    self.reranker_model = RobertaModel.from_pretrained('roberta-base')
                except Exception as e:
                    print(e)
                    self.reranker_model = RobertaModel.from_pretrained('pretrained_model/roberta_hf')
            roberta_dim = 768
                
        elif reranker_pretrain_model == 'longformer':
            from transformers import LongformerModel, LongformerTokenizer
            self.reranker_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            self.reranker_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')  
            roberta_dim = 768
        
        elif reranker_pretrain_model == 'roberta-large':
            from transformers import RobertaModel, RobertaTokenizer
            self.reranker_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            self.reranker_model = RobertaModel.from_pretrained('roberta-large')
            roberta_dim = 1024
        
        
        
        loss_method = getattr(self.cfg, "criterion", "none")
        if 'v3' in loss_method:
            from fairseq.modules import TransformerDecoderLayer
            beam_K = getattr(self.args, "length_beam_K", 10)
            
            from copy import deepcopy
            args_copy = deepcopy(self.args)
            tmp_for_restore1 = args_copy.decoder.embed_dim
            tmp_for_restore2 = args_copy.encoder.embed_dim
            args_copy.decoder.embed_dim = roberta_dim
            args_copy.encoder.embed_dim = roberta_dim
            
            self.reranker_decoder = TransformerDecoderLayer(args_copy, no_encoder_attn=False)
            
            args_copy.decoder.embed_dim = tmp_for_restore1
            args_copy.encoder.embed_dim = tmp_for_restore2
            reranker_self_attention_pos_emb = getattr(self.args, "reranker_self_attention_pos_emb", False)
            if reranker_self_attention_pos_emb:
                self.reranker_pos_emb = torch.nn.Embedding(beam_K, roberta_dim)
            
            reranker_scorer_method = getattr(self.args, 'reranker_scorer_method', 'classification')
            if reranker_scorer_method == "classification":
                self.reranker_scorer = torch.nn.Linear(roberta_dim, beam_K)
            elif reranker_scorer_method == "energy":
                self.reranker_scorer = torch.nn.Linear(roberta_dim, 1)               
        
        elif 'progressive' in loss_method:
            from fairseq.modules import TransformerDecoderLayer
            beam_K = getattr(self.args, "length_beam_K", 10)
            beam_KV = getattr(self.args, "length_beam_KV", 5)
            cat_size = beam_K * beam_KV
            
            from copy import deepcopy
            args_copy = deepcopy(self.args)
            tmp_for_restore1 = args_copy.decoder.embed_dim
            tmp_for_restore2 = args_copy.encoder.embed_dim
            args_copy.decoder.embed_dim = roberta_dim
            args_copy.encoder.embed_dim = roberta_dim
            
            self.reranker_decoder = TransformerDecoderLayer(args_copy, no_encoder_attn=False)
            
            args_copy.decoder.embed_dim = tmp_for_restore1
            args_copy.encoder.embed_dim = tmp_for_restore2
            reranker_self_attention_pos_emb = getattr(self.args, "reranker_self_attention_pos_emb", False)
            if reranker_self_attention_pos_emb:
                self.reranker_pos_emb = torch.nn.Embedding(cat_size, roberta_dim)
            
            reranker_scorer_method = getattr(self.args, 'reranker_scorer_method', 'classification')
            if reranker_scorer_method == "classification":
                self.reranker_scorer = torch.nn.Linear(roberta_dim, cat_size)
            elif reranker_scorer_method == "energy":
                self.reranker_scorer = torch.nn.Linear(roberta_dim, 1)               
        
        else:            
            reranker_score_method = getattr(self.args, 'reranker_score_method', 'ind')
            reranker_self_attention = getattr(self.args, "reranker_self_attention", False)
            
            if reranker_score_method == 'ind' and not reranker_self_attention:
                self.reranker_scorer = torch.nn.Linear(roberta_dim, 1)
            elif reranker_score_method == 'ind' and reranker_self_attention:
                self.reranker_scorer = torch.nn.Linear(roberta_dim, self.args.length_beam_K)
            else:
                raise NotImplementedError()
            
            reranker_inject_rank = getattr(self.args, "reranker_inject_rank", False)
            if reranker_inject_rank:
                num_special_tokens = getattr(self.args, "length_beam_K", 10)
                special_token_list = [f"<_{i}_>" for i in range(num_special_tokens)]
                self.reranker_tokenizer.add_tokens(special_token_list)        
                self.reranker_model.resize_token_embeddings(len(self.reranker_tokenizer))
            
            if reranker_self_attention:
                loss_method = getattr(self.cfg, "criterion", "none")
                if 'v2' in loss_method:
                    from fairseq.modules import MultiheadAttention
                    self.reranker_self_attn = MultiheadAttention(roberta_dim, 8, encoder_decoder_attention=True, dropout=0.1)            
                else:
                    from torch.nn import MultiheadAttention
                    self.reranker_self_attn = MultiheadAttention(roberta_dim, 8)          
                
            reranker_self_attention_linear = getattr(self.args, "reranker_self_attention_linear", False)
            if reranker_self_attention_linear:
                self.reranker_self_attn_q = nn.Linear(roberta_dim, roberta_dim)
                self.reranker_self_attn_k = nn.Linear(roberta_dim, roberta_dim)
                self.reranker_self_attn_v = nn.Linear(roberta_dim, roberta_dim)
            
            reranker_self_attention_pos_emb = getattr(self.args, "reranker_self_attention_pos_emb", False)
            if reranker_self_attention_pos_emb:
                self.reranker_self_attn_pos_emb = PositionalEncoding(roberta_dim)
        
        
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlatLinkDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder


                
    def extract_valid_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen

        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0)

        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))

        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len

    def restore_valid_links(self, links):
        # batch * prelen * trans_len
        batch_size, prelen, translen = links.shape
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= prelen
        valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        return res[:, :, :prelen]

    def extract_links(self, features, prev_output_tokens,
            link_positional, query_linear, key_linear, gate_linear):
        
        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")

        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens))

        features_withpos = torch.cat(links_feature_arr, dim=-1)

        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))

        if self.args.max_transition_length != -1:
            log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, prev_output_tokens.ne(self.pad))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.pad).unsqueeze(1)
            link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
            link_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        return links

    def extract_features(self, prev_output_tokens, encoder_out, rand_seed, require_links=False):
        with torch_seed(rand_seed):
            features, _ = self.decoder.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=False
            )
            # word_ins_out = self.decoder.output_layer(features)
            word_ins_out = self.decoder.output_projection(features)

            links = None
            if require_links:
                links = self.extract_links(features, \
                            prev_output_tokens, \
                            self.decoder.link_positional, \
                            self.decoder.query_linear, \
                            self.decoder.key_linear, \
                            self.decoder.gate_linear
                        )

        return word_ins_out, links
    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        rand_seed = random.randint(0, 19260817)
        # decoding
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=links)
                word_ins_out = None

        word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "nll_loss": True,
            }
        }
        ret['links'] = links

        ret["length"] = {
            "out": length_out,
            "tgt": length_tgt,
            "factor": self.decoder.length_loss_factor,
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def get_encoder_out(self, src_tokens, src_lengths):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        return encoder_out
    
    def forward_only_train_decodder(self, encoder_out, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, **kwargs):        
        rand_seed = random.randint(0, 19260817)
        # decoding
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=links)
                word_ins_out = None

        word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "nll_loss": True,
            }
        }
        ret['links'] = links

        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens_upsample_by_tokens(self, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_multiplier_by_tokens(self, src_tokens, tgt_tokens):
        length_tgt = torch.sum(tgt_tokens.ne(self.tgt_dict.pad_index), -1)
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_by_tokens(self, src_tokens, tgt_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample_by_tokens(src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier_by_tokens(src_tokens, tgt_tokens)

    def initialize_output_tokens_upsample(self, encoder_out, src_tokens):
        # length prediction
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if "length_control" in vars(self.args).get("decode_strategy", ""):
                minimal_target_length = max(max(length_tgt), getattr(self.args, 'minimal_target_length', 2)) # 
            else:
                minimal_target_length = 2
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=minimal_target_length)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens_multiplier(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample(encoder_out, src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier(encoder_out, src_tokens)

    def max_positions(self):
        if vars(self.args).get("filter_max_length", None) is not None:
            if ":" not in self.args.filter_max_length:
                a = b = int(self.args.filter_max_length)
            else:
                a, b = self.args.filter_max_length.split(":")
                a, b = int(a), int(b)
            return (a, b)
        else:
            if vars(self.args).get("src_upsample_fixed", None) is not None:
                return (self.encoder.max_positions(), self.decoder.max_positions())
            elif vars(self.args).get("src_upsample_scale", None) is not None:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.src_upsample_scale)), self.decoder.max_positions())
            else:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.length_multiplier)), self.decoder.max_positions())

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, 
                        force_decode_strategy=None, return_every_beam=False, return_all_groups=False, return_beam_and_score=False,
                        **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history
        rand_seed = random.randint(0, 19260817)

        # execute the decoder
        output_logits, links = self.extract_features(output_tokens, encoder_out, rand_seed, require_links=True)
        if self.args.max_transition_length != -1:
            links = self.restore_valid_links(links)
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
        
        decode_strategy = force_decode_strategy if force_decode_strategy is not None else self.args.decode_strategy 
        
        if decode_strategy in ["length_control", "length_control_bs"]:
            vocab_mask = torch.zeros((output_logits.size(-1))).bool().to(output_tokens.device)
            vocab_mask[self.tgt_dict.bos_index] = True
            vocab_mask[self.tgt_dict.eos_index] = True
            vocab_mask[self.tgt_dict.pad_index] = True
            output_logits.masked_fill_(vocab_mask.unsqueeze(0).unsqueeze(1), torch.tensor(-float('inf')))
        
        output_logits_normalized = output_logits.log_softmax(dim=-1)
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        if decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_strategy == "lookahead":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
            elif self.args.decode_strategy == "greedy":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen

            unpad_output_tokens = []
            for i, length in enumerate(output_length):
                last = unreduced_tokens[i][0]
                j = 0
                res = [last]
                while j != length - 1:
                    j = links_idx[i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.append(now_token)
                    last = now_token
                unpad_output_tokens.append(res)

            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
        
        elif decode_strategy in ["length_control"]:
            def get_length_by_lookahead_deocoding():
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
                unpad_output_tokens = []
                for i, length in enumerate(output_length):
                    last = unreduced_tokens[i][0]
                    j = 0
                    res = [last]
                    while j != length - 1:
                        j = links_idx[i][j]
                        now_token = unreduced_tokens[i][j]
                        if now_token != self.tgt_dict.pad_index and now_token != last:
                            res.append(now_token)
                        last = now_token
                    unpad_output_tokens.append(res)
                return torch.tensor([len(x) for x in unpad_output_tokens]).to(output_tokens), unpad_output_tokens
            
            scores = []
            indexs = []
            # batch * graph_length
            alpha_t = links[:,0]
            alpha_t += unreduced_logits[:,0].unsqueeze(1)
            batch_size, graph_length, _ = links.size()
            # alpha_t += unreduced_logits
            scores.append(alpha_t)
            
            # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
            max_length = int(2 * graph_length / self.args.src_upsample_scale)
            for i in range(max_length - 1):
                # alpha_t += unreduced_logits
                alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + links, dim = 1)
                alpha_t += unreduced_logits
                scores.append(alpha_t)
                indexs.append(index)

            # max_length * batch * graph_length
            indexs = torch.stack(indexs, dim = 0)
            scores = torch.stack(scores, dim = 0)
            # link_last = torch.gather(links, -1, (output_length - 1).view(batch_size, 1, 1).repeat(1, graph_length, 1)).view(1, batch_size, graph_length)
            # scores += link_last # NOTE: what is this line for?

            # max_length * batch
            scores, max_idx = torch.max(scores, dim = -1)
            lengths = torch.arange(max_length).unsqueeze(-1).repeat(1, batch_size) + 1
            length_penalty = (lengths ** self.args.decode_viterbibeta).cuda(scores.get_device())
            scores = scores / length_penalty
            max_score, pred_length = torch.max(scores, dim = 0)
            
            dp_pred_length = pred_length.clone()
            # pred_length[:] = 10
            specified_length_fixed = getattr(self.args, 'specified_length_fixed', None)
            specified_length_ratio = getattr(self.args, "specified_length_ratio", None)
            
            not_force_length = getattr(self.args, "not_force_length", None)
            
            if specified_length_fixed is not None:
                new_length = torch.tensor(specified_length_fixed).expand_as(encoder_out['src_lengths'][0]).view(-1).to(pred_length) - 1
            elif specified_length_ratio is not None:
                new_length = torch.floor(encoder_out['src_lengths'][0] * specified_length_ratio).view(-1).to(pred_length)

            if not_force_length:
                supposeed_predicted_length, supposeed_predicted_tokens = get_length_by_lookahead_deocoding()
                not_force_length_cond1 = supposeed_predicted_length < new_length
                not_force_length_cond2 = encoder_out['src_lengths'][0].view(-1) < new_length
                not_force_length_cond = torch.logical_and(not_force_length_cond1, not_force_length_cond2)
                pred_length[~not_force_length_cond] = new_length[~not_force_length_cond]
            else:
                pred_length[:] = new_length   
            
            pred_length = pred_length + 1

            initial_idx = torch.gather(max_idx, 0, (pred_length - 1).view(1, batch_size)).view(batch_size).tolist()
            
            # initial_idx = torch.ones(alpha_t.size(0)).to(max_idx) * (alpha_t.size(1)-1)
            
            unpad_output_tokens = []
            indexs = indexs.tolist()
            pred_length = pred_length.tolist()
            for i, length in enumerate(pred_length):
                j = initial_idx[i]
                last = unreduced_tokens[i][j]
                res = [last]
                for k in range(length - 1):
                    j = indexs[length - k - 2][i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.insert(0, now_token)
                    last = now_token
                unpad_output_tokens.append(res)

            if not_force_length:
                merged_output = []
                for i, is_force in enumerate(not_force_length_cond):
                    if is_force:
                        merged_output.append(unpad_output_tokens[i])
                    else:
                        merged_output.append(supposeed_predicted_tokens[i])
                unpad_output_tokens = merged_output
                
            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
        
        elif decode_strategy in ["length_control_bs"]:                          
                    
            def merge(G1, G2, not_rerank=False, j=None, encoder_out=None):
                # bsz = G1.beams.size(0)
                # device = G1.beams.device
                G1_score_cp = G1.score.clone()
                # create pre_match_cond_beam here
                
                
                for k in range(K):
                    # Original:
                    # G1_beam = G1.beams[:, k, :].unsqueeze(1)
                    # pre_match_cond_beam = (G2.beams == G1_beam).all(dim=2) 
                    # pre_match_cond_beam_ext = torch.cat((torch.zeros((bsz, 1), device=device), pre_match_cond_beam.int()), dim=1)
                    # # NOTE: change to only match the first one
                    # index = torch.stack((torch.arange(bsz, device=device), pre_match_cond_beam_ext.argmax(-1)), dim=1)
                    # match_cond_beam = torch.zeros_like(pre_match_cond_beam_ext)
                    # match_cond_beam[index[:,0], index[:,1]] = 1
                    # match_cond_beam = match_cond_beam.bool()[:, 1:]
                    G1_beam = G1.beams[:, k, :].unsqueeze(1)
                    pre_match_cond_beam = (G2.beams == G1_beam).all(dim=2) #bsz * K
                    pre_match_cond_beam_ext[:, 1:] = pre_match_cond_beam.int() # bsz * K+1
                    # NOTE: change to only match the first one
                    index = pre_match_cond_beam_ext.argmax(-1)
                    match_cond_beam_init.fill_(0)
                    match_cond_beam_init[index_val, index] = 1
                    match_cond_beam = match_cond_beam_init.bool()[:, 1:]
                    
                    match_cond_in_G2 = match_cond_beam.unsqueeze(1).expand(-1, L, -1)        
                    G2.score[G2.score == -float("inf")] = 0
                    G1.score[G1.score == -float("inf")] = 0
                    
                    match_cond_in_G1 = torch.zeros(match_cond_in_G2.size(0), 1 , match_cond_in_G2.size(2), device=match_cond_in_G2.device) # where else can we put?
                    match_cond_in_G1[pre_match_cond_beam.any(dim=1), :, k] = 1
                    match_cond_in_G1 = match_cond_in_G1.expand(-1, L, -1).bool()
                    G2.score[match_cond_in_G2] = G2.score[match_cond_in_G2] + G1.score[match_cond_in_G1]
                    
                    G1_score_cp[match_cond_in_G1] = -float("inf")
                    
                    G1.score[G1.score == 0] = -float("inf")
                    G2.score[G2.score == 0] = -float("inf")

                if not not_rerank:
                    G2.score = torch.cat((G2.score, G1_score_cp), dim=2)
                    G2.beams = torch.cat((G2.beams, G1.beams), dim=1)
                   
                    G2_score_for_rank = G2.score

                    dat_score = G2_score_for_rank.logsumexp(1).exp()
                    
                    if G2.beams.shape[2] > 1 and G2.rerank is not None:
                        temp_rerank = torch.cat((G2.rerank, G1.rerank), dim=1)
                        rerank_lambda = getattr(self.args, 'rerank_lambda', 0.01)
                        score_for_rank = dat_score + rerank_lambda * temp_rerank ## 80 x 20?
                    else:
                        score_for_rank = dat_score
                 
                    merge_idx = score_for_rank.topk(K)[1]      
                    
                    G2.score = G2.score.gather(2, merge_idx.unsqueeze(1).expand(-1, L, -1))
                    G2.beams = G2.beams.gather(1, merge_idx.unsqueeze(2).expand(-1, -1, G2.beams.size(2)))
                    if G2.beams.shape[2] > 1 and G2.rerank is not None:
                        G2.rerank = temp_rerank.gather(1, merge_idx)

            # Main decoding algorithm                   
            P = output_logits_normalized 
            E = links
            bsz, L, _ =  E.size()
            K = getattr(self.args, "length_beam_K", 10)
            K_vocab = getattr(self.args, "length_beam_KV", 1)
            device = E.device

            pre_match_cond_beam_ext = torch.zeros(bsz, K+1, device=device)
            match_cond_beam_init = torch.zeros(bsz, K+1, device=device)
            index_val = torch.arange(bsz, device=device)
            
            # Predicted length
            specified_length_fixed = getattr(self.args, 'specified_length_fixed', None)
            specified_length_ratio = getattr(self.args, "specified_length_ratio", None)
            
            if specified_length_fixed is not None:
                new_length = torch.tensor(specified_length_fixed).expand_as(encoder_out['src_lengths'][0]).view(-1).to(device) - 1
            elif specified_length_ratio is not None:
                new_length = torch.floor(encoder_out['src_lengths'][0] * specified_length_ratio).view(-1).to(device).int()
            elif getattr(self.args, "char_control_method", None) == "bucket":
                new_length = torch.tensor(10).expand_as(encoder_out['src_lengths'][0]).view(-1).to(device)
            else:
                new_length = torch.tensor(1).expand_as(encoder_out['src_lengths'][0]).view(-1).to(device) - 1

                
            pred_length = new_length.view(-1)   

            if getattr(self.args, "char_control_method", None) == "greedy":
                beam_len_list = []
                for l in range(L):
                    if G[l][-1].beams is not None:
                        beam_len = [sum([self.id2len[int(_id)] for _id in item[0]]) for item in G[l][-1].beams]
                        beam_len_list.append(beam_len)
                beam_len_tensor = torch.tensor(beam_len_list)
                beam_len_tensor[beam_len_tensor>50] = 0
                pred_length = beam_len_tensor.argmax(0)


            # Decoding algorithm 
            G = [[Beams(bsz, L, K, E.device) for i in range(L)] for j  in range(L)]

            topk_k_word_score, top_k_word_idx = P.topk(K_vocab, -1)
            topk_k_beam_score = topk_k_word_score + E[:, 0, :].unsqueeze(2)

            P_topK_score, P_topK_idx = P.topk(K_vocab, -1)
                
            for j in range(1, L):
                G[0][j].init_score(topk_k_beam_score[:, j, :], j, top_k_word_idx[:, j, :])
                if j > 1:
                    merge(G[0][j-1], G[0][j], not_rerank=False, j=j, encoder_out=encoder_out)
            
            # max(pred_length) + 2
            for i in range(1, L):
                for j in range(i + 1, L):
                    this_link = E[:, :, j].unsqueeze(2)
                    
                    TScore = log_bmm(G[i-1][j-1].score.permute(0,2,1), this_link)
                    
                    G[i-1][j-1].beams

                    # evalualte K * K_vocab amount of rerank scores                   

                    rerank_method = getattr(self.args, "rerank_method", "none")
                    rerank_lambda = getattr(self.args, 'rerank_lambda', 0.01)
                    
                    reranker_progressive_method = getattr(self.args, "reranker_progressive_method", 'none')
                    if reranker_progressive_method != 'none':
                        rerank_method = reranker_progressive_method
                                        
                    rereanking_score = 0
                    new_beams = None
                    pre_rereanking_score = None
                        
                    next_prob_score = P_topK_score[:,j,:].unsqueeze(1) + TScore
                    next_score_for_rerank = next_prob_score + rerank_lambda * rereanking_score
                    _, next_topK_idx = (next_score_for_rerank).reshape(bsz, -1).topk(K)
                    
                    if rerank_method != "none" and 'final' not in rerank_method and 'progressive_bert_train' != rerank_method:
                        top_rr_score = rereanking_score.reshape(bsz, -1).gather(1, next_topK_idx)
                    else:
                        top_rr_score = None
                    
                    topK_pre_beams = G[i-1][j-1].beams.gather(1, (next_topK_idx // K_vocab).unsqueeze(2).expand(-1, -1, G[i-1][j-1].beams.size(2)))
                    topK_next_prob_score = next_prob_score.reshape(bsz, -1).gather(1, next_topK_idx)
                    
                    new_words = P_topK_idx[:,j,:].gather(1, next_topK_idx % K_vocab).unsqueeze(2)

                    G[i][j].beams = torch.cat((topK_pre_beams, new_words), dim=2)
                    G[i][j].rerank = top_rr_score
                    G[i][j].all_beams = new_beams
                    G[i][j].all_rerank = pre_rereanking_score
                    
                    G[i][j].init_score(topK_next_prob_score, j)
                    if j > i + 1:
                        merge(G[i][j-1], G[i][j], not_rerank=False, j=j, encoder_out=encoder_out)
                    
            def _postprocess(tokens):
                _toks = [[v for i, v in enumerate(_a_toks) if v != _a_toks[i - 1]] for _a_toks in tokens]
                max_len = max([len(x) for x in _toks])
                _tokens = [_a_toks + [self.tgt_dict.pad_index] * (max_len - len(_a_toks)) for _a_toks in _toks]
                hyp = torch.tensor(_tokens, device=E.device)
                return hyp
            
            # score_list = []
            # for i in range(L):
            #     one_score = G[i][-1].score.logsumexp(1)
            #     score_list.append(one_score)
            # len_pen = torch.tensor(range(L), device=E.device) + 1
            # len_pen = len_pen ** 0.9
            # len_pen = len_pen.unsqueeze(0).unsqueeze(2).expand(bsz, -1, -1)
            # pen_score = (torch.stack(score_list, dim=1) / len_pen)
            # pred_length = pen_score.max(-1)[0].argmax(-1)
            
            final_step_rerank = True if 'final' in rerank_method else False
            unpad_output_tokens = []
            reranking_score_list = []
            for i, one_pred_length in enumerate(pred_length):
                # one_pred_length = min(len(G) - 2, pred_length.int())
                if not return_every_beam:
                    if not final_step_rerank:
                        unpad_output_tokens.append(G[one_pred_length][-1].beams[i, 0])
                    else:
                        new_beams = G[one_pred_length][-1].beams[i].unsqueeze(0)
                        if rerank_method == "final_bert_v3":
                            pre_rereanking_score = reranker_bert_v3(encoder_out['src_tokens'][0][i].unsqueeze(0), new_beams, self, device)
                            
                        if return_beam_and_score:
                            unpad_output_tokens.append(_postprocess(G[one_pred_length][-1].beams[i]))                
                            reranking_score_list.append(pre_rereanking_score)
                        else: # return the best
                            pre_rereanking_score_idx = pre_rereanking_score.argmax()
                            unpad_output_tokens.append(G[one_pred_length][-1].beams[i, pre_rereanking_score_idx])
                else:  
                    unpad_output_tokens.append(G[one_pred_length][-1].beams[i])
                    prob_rereanking_score = G[one_pred_length][-1].score[i].logsumexp(0)
                    reranking_score_list.append(prob_rereanking_score)
                    
            if return_beam_and_score:
                return decoder_out._replace(
                    output_tokens=unpad_output_tokens,
                    output_scores=reranking_score_list,
                    attn=None,
                    history=history,
                )
                        
            if return_every_beam:
                return unpad_output_tokens, reranking_score_list
            
            if return_all_groups:
                return G, pred_length
            
            output_tokens = _postprocess(unpad_output_tokens)

            
        elif self.args.decode_strategy == "beamsearch":
            batch_size, prelen, _ = links.shape

            assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

            top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
            dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
            dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

            nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
            logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
            idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
            logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

            rearange_idx = logits_idx.sort(dim=-1)[1]
            dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            
            dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
            nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
            logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
            output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())

            res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0
            )
            output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=torch.full(output_tokens.size(), 1.0, device=output_tokens.device),
            attn=None,
            history=history,
        )


class GlatLinkDecoder(NATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(args)

    def init_link_feature(self, args):
        links_feature = self.args.links_feature.split(":")
        links_dim = 0
        if "feature" in links_feature:
            links_dim += args.decoder_embed_dim
        if "position" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, True)
            links_dim += args.decoder_embed_dim
        elif "sinposition" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, False)
            links_dim += args.decoder_embed_dim
        else:
            self.link_positional = None

        self.query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
        self.key_linear = nn.Linear(links_dim, args.decoder_embed_dim)
        self.gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)

    @staticmethod
    def add_args(parser):
        pass

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_base"
)
def base_architecture2(args):
    base_architecture(args)