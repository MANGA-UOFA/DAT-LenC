# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from json import decoder
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.fairseq_decoder import FairseqDecoder
import torch
# from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from fs_plugins.models.nat_long import NATransformerModel


import logging
import random
from lunanlp import torch_seed


logger = logging.getLogger(__name__)



@register_model("glat")
class Glat(NATransformerModel):
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

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
        parser.add_argument(
            "--restore-decoder-from",
            default="off",
            action="store",
        )
        parser.add_argument(
            '--dropout-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal-end-ratio',
            type=float,
            default=0
        )
        parser.add_argument(
            '--no-glat-input',
            action='store_true',
            default=False
        )


    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, train_ratio=None, **kwargs
    ):
        if train_ratio is not None:
            self.encoder.train_ratio = train_ratio
            self.decoder.train_ratio = train_ratio
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
            if "context_p" in glat:
                with torch.no_grad():
                    with torch_seed(rand_seed):
                        word_ins_out = self.decoder(
                            normalize=False,
                            prev_output_tokens=prev_output_tokens,
                            encoder_out=encoder_out,
                        )
                    pred_tokens = word_ins_out.argmax(-1)
                    nonpad_positions = ~tgt_tokens.eq(self.pad)
                    same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
                    seq_lens = (nonpad_positions).sum(1)
                    keep_prob = ((seq_lens - same_num) / seq_lens * glat['context_p']).unsqueeze(-1)
                    # keep: True, drop: False
                    keep_word_mask = (
                            torch.rand(prev_output_tokens.shape, device=word_ins_out.device) < keep_prob).bool()
                    glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask,
                                                                             0) + tgt_tokens.masked_fill(
                        ~keep_word_mask, 0)
                    
                    glat_tgt_tokens = tgt_tokens.masked_fill(keep_word_mask, self.pad)
                    
                    tgt_tokens = glat_tgt_tokens
                    if self.args.no_glat_input is False:
                        prev_output_tokens = glat_prev_output_tokens

                    glat_info = {
                        "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                        "glat_context_p": glat['context_p'],
                        "glat_keep": keep_prob.mean().item()
                    }


        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            }
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

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

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

    def analysis(self, sample):
        import torch 

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)
        
        tested_id = 1412 # 1412, 5920, 6005, 1333, 6003, 6451, 4686, 1381, 3264, 2934, 5
        
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        tgt_lengths = torch.sum(sample['target'].ne(self.tgt_dict.pad()), dim=-1)
        
        tgt_tokens_decoded = [ [self.tgt_dict[x] for x in line] for line in sample['target']]
        
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        
        prev_output_tokens = _full_mask(sample['target'])
        
        init_out_logits = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        
        _scores, _tokens = init_out_logits.max(-1)    
        
        from torch.distributions import Categorical
        self_entropy = Categorical(logits=init_out_logits).entropy()
        
        output_tokens = prev_output_tokens
        
        max_len = output_tokens.size(1)
        
        # STEP 1: Get the mutural information list
        cond_entorpy_list = []
        for i in range(max_len):
            tem_output_masks = output_tokens.clone()
            tem_output_masks[:, i] = sample['target'][:, i]
            
            _tmp_scores = self.decoder(
                normalize=False,
                prev_output_tokens=tem_output_masks,
                encoder_out=encoder_out,
            )  
            entropy = Categorical(logits=_tmp_scores).entropy()
            cond_entorpy_list.append(entropy)            
            
        cond_entorpy = torch.stack(cond_entorpy_list, dim=-1)
        
        output_masks = output_tokens.eq(self.unk) 
        mi_mask = torch.bmm(output_masks.unsqueeze(2).to(cond_entorpy), output_masks.unsqueeze(1).to(cond_entorpy)).bool()
        
        # self_unmask = (~(torch.eye(mi_mask.size(2)).unsqueeze(0).repeat(mi_mask.size(0), 1, 1).bool())).to(cond_entorpy).bool()
        # mi_mask = torch.logical_and(self_unmask, mi_mask)
        
        mi_matrix = torch.zeros_like(cond_entorpy)
        
        # mi_matrix = mi_matrix.masked_fill(~mi_mask, float(-0.1))    
        
        mi_matrix = torch.zeros_like(cond_entorpy)
        for i in range(max_len):
            for j in range(max_len):
                mi_matrix[:, i, j] = self_entropy[:, i] - cond_entorpy[:, j, i]

                
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        ignore_mi_mask = torch.logical_or((sample['target'] == self.pad), (sample['target'] == self.bos))
        ignore_mi_mask = torch.logical_or(ignore_mi_mask, (sample['target'] == self.eos))
        ignore_mi_mask = torch.bmm(ignore_mi_mask.unsqueeze(2).to(cond_entorpy), ignore_mi_mask.unsqueeze(1).to(cond_entorpy)).bool()
        
        for idx, one_mat in enumerate(mi_matrix):
            one_heat_map = one_mat.cpu().numpy()
            # one_heat_map = 
            this_length = tgt_lengths[idx]
            one_heat_map = np.absolute(one_heat_map[1:this_length-1, 1:this_length-1])
            real_tokens = tgt_tokens_decoded[idx][1:this_length-1]
            plt.figure()
            colormap = sns.color_palette("Greens")
            ax = sns.heatmap(one_heat_map, xticklabels=real_tokens, yticklabels=real_tokens, cmap=colormap)
            ax.xaxis.set_label_position('top')
            sample_id = sample['id'][idx]    
            plt.savefig(f'heat_map/glat/{sample_id}.png')
        exit(0)
        

@register_model_architecture(
    "glat", "nat_glat"
)
def base_architecture0(args):
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
    "glat", "glat_base"
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
    "glat", "glat_big"
)
def big_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture(
    "glat", "glat_16e6d"
)
def glat_16e6d_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)

