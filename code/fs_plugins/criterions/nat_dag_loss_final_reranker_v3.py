# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
import logging
from functools import reduce
import numpy as np
from typing import Union, Tuple, Optional
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.autograd import Function
from ..custom_ops import dag_loss, dag_best_alignment, dag_logsoftmax_gather_inplace, torch_dag_loss, torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace

from .utilities import parse_anneal_argument, get_anneal_value

logger = logging.getLogger(__name__)

########### gpu use tracker ###########
# import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

import nltk 
from rouge_score import rouge_scorer

import random

DEBUG = False

chencherry = nltk.translate.bleu_score.SmoothingFunction()
smoothing_function=chencherry.method1
bleu_fn = nltk.translate.bleu_score.sentence_bleu 
# bleu_fn([one_tgt_str], one_beam_str, smoothing_function=smoothing_function, weights=(0.5, 0.5))
rouge_model = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])


def reranker_rouge(_ref, _hyp, model=None):
    ref = [int(x) for x in _ref]
    hyp = [int(x) for x in _hyp]
    ref = ' '.join([model.id2word[x]  for x in ref if x not in [0, 1, 2]])
    hyp = ' '.join([model.id2word[x]  for x in hyp if x not in [0, 1, 2]])
    # ref2 = ' '.join([str(x)  for x in _ref if x not in [0, 1, 2]])
    # hyp2 = ' '.join([str(x)  for x in _hyp if x not in [0, 1, 2]])
    return rouge_model.score(hyp, ref)['rouge1'].fmeasure

def reranker_bleu(ref, hyp, model=None):
    ref = [int(x) for x in ref]
    hyp = [int(x) for x in hyp]
    ref_tokens = [str(x)  for x in ref if x not in [0, 1, 2]]
    hyp_tokens = [str(x)  for x in hyp if x not in [0, 1, 2]]
    reranking_score = nltk.translate.bleu_score.sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function, weights=(0.25, 0.25, 0.25, 0.25))
    return reranking_score


def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    # mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        # mask[i, :length] = 1
    return out_tensor #, mask


@register_criterion("nat_dag_final_reranking_loss_v3")
class NATDAGFinetuneLoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        assert cfg.label_smoothing == 0, "DAG does not support label smoothing"
        self.glance_strategy = cfg.glance_strategy
        self._glat_p_anneal_params = parse_anneal_argument(cfg.glat_p)

        self.set_update_num(0)
        
        if DEBUG:
            self.dropout = torch.nn.Dropout(0.0)
            self.output_file_for_debug = open("inferece_scores.txt", 'w')
        else:
            self.dropout = torch.nn.Dropout(0.1)
        
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0)
        parser.add_argument("--glat-p", type=str, default="0")
        parser.add_argument("--glance-strategy", type=str, default=None)
        parser.add_argument("--no-force-emit", action="store_true")

        parser.add_argument("--torch-dag-logsoftmax-gather", action="store_true")
        parser.add_argument("--torch-dag-best-alignment", action="store_true")
        parser.add_argument("--torch-dag-loss", action="store_true")
        
        # finetune args
        parser.add_argument("--finetune-with-gt", action="store_true")
        parser.add_argument("--finetune-gt-weight", type=float, default=1.0)
        parser.add_argument("--finetune-margin", type=float, default=0.1)
        
        parser.add_argument("--reranker-temperature", type=float, default=10)
        parser.add_argument("--reranker-score-method", type=str, default='ind')
        parser.add_argument("--reranker-target-dist", type=str, default='rouge')
        parser.add_argument("--reranker-loss-reweight", action="store_true", default=False)
        parser.add_argument("--reranker-fix-bert", action="store_true", default=False)

    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = utils.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "ntokens": outputs.shape[0], "loss_nofactor": loss_nofactor}

    def _compute_dag_loss(self, outputs, output_masks, targets, target_masks, links, label_smoothing=0.0, name="loss",
                factor=1.0, matchmask=None, keep_word_mask=None, model=None):

        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        if self.cfg.torch_dag_logsoftmax_gather:
            outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        else:
            outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)

        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()

        # calculate
        if self.cfg.torch_dag_loss:
            if model.args.max_transition_length != -1:
                links = model.restore_valid_links(links)
            loss_result = torch_dag_loss(match_all, links, output_length, target_length)
        else:
            assert model.args.max_transition_length != -1, "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
            loss_result = dag_loss(match_all, links, output_length, target_length)

        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()
        
        loss_none_resuded = - (loss_result / target_length)
        loss = loss_none_resuded.mean()
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.tgt_dict.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences, "loss_none_resuded": loss_none_resuded}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def set_update_num(self, update_num):
        self.glat_p = get_anneal_value(self._glat_p_anneal_params, update_num)

    def forward(self, model, sample, reduce=True, fast_eval=False, is_training=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # import gc
        # gc.collect()
        if SHOW_MEMORY_USE:
            print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
            gpu_tracker.clear_cache()
        # gpu_tracker.track()

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]

        if SHOW_MEMORY_USE:
            print(sample["net_input"]["src_tokens"].shape[0], sample["net_input"]["src_tokens"].shape[1], tgt_tokens.shape[1], file=sys.stderr, end=" ")

        if sample.get("update_num", None) is not None: # in training            
            self.set_update_num(sample['update_num'])

        if self.glat_p == 0:
            glat = None
        else:
            glat = {
                "context_p": max(self.glat_p, 0),
                "require_glance_grad": False
            }

        def glat_function(model, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=None):
            batch_size, prelen, _ = links.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)

            pred_tokens = word_ins_out.argmax(-1)
            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            match = match.transpose(1, 2)
            
            if self.cfg.torch_dag_best_alignment:
                if model.args.max_transition_length != -1:
                    links = model.restore_valid_links(links)
                path = torch_dag_best_alignment(match, links, output_length, target_length)
            else:
                assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                path = dag_best_alignment(match, links, output_length, target_length) # batch * prelen

            predict_align_mask = path >= 0
            matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=match.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
            oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
            same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)
       
            if self.glance_strategy is None:
                keep_prob = ((target_length - same_num) / target_length * glat['context_p']).unsqueeze(-1) * predict_align_mask.float()

            elif self.glance_strategy in ['number-random']:
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = ((target_length - same_num) * glat['context_p'] + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            elif self.glance_strategy == "cmlm":
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = (target_length * torch.rand_like(target_length, dtype=torch.float) + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
            
            glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
            glat_tgt_tokens = tgt_tokens
        
            glat_info = {
                "glat_accu": (same_num.sum() / target_length.sum()).detach(),
                "glat_context_p": glat['context_p'],
                "glat_keep": keep_prob.mean().detach(),
                "matchmask": matchmask,
                "keep_word_mask": keep_word_mask,
                "glat_prev_output_tokens": glat_prev_output_tokens,
            }

            return glat_prev_output_tokens, glat_tgt_tokens, glat_info

        losses = []

        model.eval()
        
        device = tgt_tokens.device

        with torch.no_grad():
            encoder_out = model.get_encoder_out(src_tokens, src_lengths)
            decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)
            kwargs = {}
            kwargs['target'] = tgt_tokens
            beam_results, beam_scores = model.forward_decoder(decoder_out, encoder_out, decoding_format=None,
                                                                force_decode_strategy="length_control_bs", 
                                                                return_every_beam=True, **kwargs)

        src_tokens_decoded = [[model.id2word[int(x)] for  x in one_sample if int(x) not in [1]] for one_sample in src_tokens]
        tgt_tokens_decoded = [[model.id2word[int(x)] for  x in one_sample if int(x) not in [1]] for one_sample in tgt_tokens]
                    
        all_src_list = []
        all_tgt_list = []
        all_label_list = []
        all_rank_list = []
        all_argmax_list = []
        _reranker_label_norm_tmp = getattr(self.cfg, "reranker_temperature", 10)
        dtype = encoder_out['encoder_out'][0].dtype
        
        for i, (src, one_sample) in enumerate(zip(src_tokens_decoded, beam_results)):
            reranker_target_dist = getattr(self.cfg, "reranker_target_dist", 'rouge')
            if reranker_target_dist == 'rouge':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)    
                all_rank_list.append(label_dist.clone())
                label_dist = ((label_dist + 0.001).log() * _reranker_label_norm_tmp).softmax(-1)
                
            elif reranker_target_dist == 'bleu':
                rerank_score = [reranker_bleu(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                label_dist = ((label_dist + 0.001).log() * _reranker_label_norm_tmp).softmax(-1)

            elif reranker_target_dist == 'hard':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_label = label_dist.argmax(-1)
                hard_label_dist = torch.zeros_like(label_dist)
                hard_label_dist[best_label] = 1.0
                label_dist = hard_label_dist
                
            elif reranker_target_dist == 'hard_ls':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_label = label_dist.argmax(-1)
                all_argmax_list.append(best_label)
                mu = 0.1
                num_label = len(label_dist)
                hard_label_dist = torch.ones_like(label_dist) * mu / (num_label - 1) 
                hard_label_dist[best_label] = 1 - mu
                label_dist = hard_label_dist
            
            elif reranker_target_dist == 'bleu_hard_ls':
                rerank_score = [reranker_bleu(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_label = label_dist.argmax(-1)
                all_argmax_list.append(best_label)
                mu = 0.1
                num_label = len(label_dist)
                hard_label_dist = torch.ones_like(label_dist) * mu / (num_label - 1) 
                hard_label_dist[best_label] = 1 - mu
                label_dist = hard_label_dist
            
            elif reranker_target_dist == 'multi_hard':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_score = label_dist.max(-1)[0]
                num_of_best = sum(label_dist == best_score).to(best_score)

                hard_label_dist = torch.zeros_like(label_dist)
                hard_label_dist[label_dist == best_score] = 1.0 / sum(label_dist == best_score).to(best_score)
                label_dist = hard_label_dist
            
            elif reranker_target_dist == 'multi_hard_ls':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_score = label_dist.max(-1)[0]
                mu = 0.1
                num_label = len(label_dist)
                num_of_best = sum(label_dist == best_score).to(best_score)
                hard_label_dist = torch.ones_like(label_dist) * mu / (num_label - num_of_best)
                hard_label_dist[label_dist == best_score] = (1.0 - mu) / num_of_best
                label_dist = hard_label_dist
            
            elif reranker_target_dist == 'prefer_ls':
                rerank_score = [reranker_rouge(tgt_tokens[i], one_beam, model) for one_beam in  one_sample]
                label_dist = torch.tensor(rerank_score, device=device, dtype=dtype)
                all_rank_list.append(label_dist.clone())
                best_label = label_dist.argmax(-1)
                all_argmax_list.append(best_label)
                mu = 0.1
                num_label = len(label_dist)
                rerank_score_tensor = torch.tensor(rerank_score, device=device, dtype=dtype) + 0.01 # avoid divide by zero
                rerank_score_tensor[best_label] = 0
                hard_label_dist = rerank_score_tensor / rerank_score_tensor.sum() * mu  
                hard_label_dist[best_label] = 1 - mu
                label_dist = hard_label_dist
                        
            all_src_list.append(' '.join(src))
            all_tgt_list.append([' '.join(['<s>'] + [model.id2word[int(x)] for x in beam if int(x) not in [0, 1, 2]] + ['</s>']) for  beam in one_sample])
            
            all_label_list.append(label_dist.clone())
                
        
        reranker_fix_bert = getattr(self.cfg, 'reranker_fix_bert', False)
        if not reranker_fix_bert and is_training:
            model.reranker_model.train()
        else:
            model.reranker_model.eval()
        
        if is_training:
            model.reranker_scorer.train()
            model.reranker_decoder.train() 

        reranker_self_attention_pos_emb = getattr(self.cfg, "reranker_self_attention_pos_emb", False)
        if is_training and reranker_self_attention_pos_emb:
              model.reranker_pos_emb.train()
        
        # define a function generates continous integers from 0 to n-1, where the repeated times of each integer is the corresponding value in input tensor
        def repeat_interleave(input):
            return torch.repeat_interleave(torch.arange(input.size(0), device=device), input)
        
        # define a function that stack list of 2-dim tensors on 0 dim
        # the 2nd dim are the same, the 1st dim are different
        # pad them according the tensor of max length on first dim
        def stack_pad_2d(input):
            max_len = max([x.size(0) for x in input])
            pad_input = [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in input]
            return torch.stack(pad_input)     
        
        def stack_pad_1d(input):
            max_len = max([x.size(0) for x in input])
            pad_input = [F.pad(x, (0, max_len - x.size(0))) for x in input]
            return torch.stack(pad_input)
                  
        # a function makes mask from a list of lenghts
        def make_mask(lengths):
            lengths = torch.tensor(lengths, device=device)
            return torch.arange(lengths.max(), device=device)[None, :] < lengths[:, None]
        
        allow_bert_grad = not reranker_fix_bert
        
        src_input = model.reranker_tokenizer(all_src_list, padding=True, return_tensors='pt', truncation=True).to(device)
        src_mask = 1 - src_input['attention_mask'].float()
        with torch.set_grad_enabled(allow_bert_grad):
            src_rep = model.reranker_model(**src_input).last_hidden_state
        if is_training:
            src_rep = self.dropout(src_rep)
        
        
        all_beam_len_list = []
        all_beam_rep_list = []
        all_beam_pos_emb_input_list = []
        all_beam_sep_len_list = []
        
        for one_beam in all_tgt_list:
            one_beam_input = model.reranker_tokenizer(one_beam, padding=True, return_tensors='pt', truncation=True).to(device)
            with torch.set_grad_enabled(allow_bert_grad):
                one_beam_rep = model.reranker_model(**one_beam_input).last_hidden_state
            if is_training:
                one_beam_rep = self.dropout(one_beam_rep)
                
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
        
        if reranker_self_attention_pos_emb:
            all_beam_rep += model.reranker_pos_emb(all_beam_pos_emb_input)
            
        fea_atten_out = model.reranker_decoder(
            x = all_beam_rep.permute(1, 0, 2),
            encoder_out = src_rep.permute(1, 0, 2),
            encoder_padding_mask = src_mask.to(dtype),
            self_attn_padding_mask = all_beam_mask.to(dtype)
        )[0]
        
        reranker_scorer_method = getattr(self.cfg, 'reranker_scorer_method', 'classification')
        
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
                           
        _label = torch.stack(all_label_list, dim=0)
        
        loss = torch.nn.KLDivLoss(reduction="none")(pred_rank, _label).sum(-1).mean()
        
        _rank = torch.stack(all_rank_list, dim=0)
        
        top_1_acc = sum(pred_rank.argmax(-1) == _rank.argmax(-1))
        top_2_acc = sum([a_pred_rank.argmax(-1) in a_label.topk(2)[1] for a_pred_rank, a_label in zip(pred_rank, _rank)])
        top_3_acc = sum([a_pred_rank.argmax(-1) in a_label.topk(3)[1] for a_pred_rank, a_label in zip(pred_rank, _rank)])
        greedy_acc = sum(_rank.argmax(-1) == 0)
        greedy_top2_acc = sum([0 in a_label.topk(2)[1] for a_pred_rank, a_label in zip(pred_rank, _rank)])
        greedy_top3_acc = sum([0 in a_label.topk(3)[1] for a_pred_rank, a_label in zip(pred_rank, _rank)])
            
        # NOTE: For debug
        # if DEBUG and not is_training:
        #     for one_src, one_predict, one_label, one_rank in zip(all_src_list, pred_rank, all_label_list, _rank):
        #         self.output_file_for_debug.write(one_src + '\t' + ' '.join([str(float(x)) for x in one_predict]) + '\t' + ' '.join([str(float(x)) for x in one_label]) + '\t' + ' '.join([str(float(x)) for x in one_rank]) + '\n')

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "top_1_acc": top_1_acc,
            "top_2_acc": top_2_acc,
            "top_3_acc": top_3_acc,
            "greedy_acc": greedy_acc,
            "greedy_top2_acc": greedy_top2_acc,
            "greedy_top3_acc": greedy_top3_acc,
            "ntokens": 1,
            "nsentences": len(all_label_list),
            "invalid_nsentences": 1,
            "tokens_perc": 1,
            "sentences_perc": 1,
            "sample_size": 1                   
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss_nofactor"])
                if reduce
                else l["loss_nofactor"]
            )

        # gpu_tracker.track()
        if DEBUG:
            loss =  torch.tensor(0.0).to(device)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )  # each batch is 1
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nvalidtokens = sum(log.get('nvalidtokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        invalid_nsentences = sum(log.get('invalid_nsentences', 0) for log in logging_outputs)
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))
        top1_acc = utils.item(sum(log.get("top_1_acc", 0) for log in logging_outputs))
        top2_acc = utils.item(sum(log.get("top_2_acc", 0) for log in logging_outputs))
        top3_acc = utils.item(sum(log.get("top_3_acc", 0) for log in logging_outputs))
        greedy_acc = utils.item(sum(log.get("greedy_acc", 0) for log in logging_outputs))
        greedy_top2_acc = utils.item(sum(log.get("greedy_top2_acc", 0) for log in logging_outputs))
        greedy_top3_acc = utils.item(sum(log.get("greedy_top3_acc", 0) for log in logging_outputs))
        
        res = {
            "ntokens": utils.item(ntokens),
            "nsentences": utils.item(nsentences),
            "nvalidtokens": utils.item(nvalidtokens),
            "invalid_nsentences": utils.item(invalid_nsentences),
            'tokens_perc': utils.item(nvalidtokens / ntokens),
            'sentences_perc': 1 - utils.item(invalid_nsentences / nsentences),
            'top1_acc': utils.item(top1_acc / nsentences),
            'top2_acc': utils.item(top2_acc / nsentences),
            'top3_acc': utils.item(top3_acc / nsentences),
            'greedy_acc': utils.item(greedy_acc / nsentences),
            'greedy_top2_acc': utils.item(greedy_top2_acc / nsentences),
            'greedy_top3_acc': utils.item(greedy_top3_acc / nsentences)
        }
        res["loss"] = loss / sample_size
        # res["glat_acc"] = glat_acc / sample_size
        # res["glat_keep"] = glat_keep / sample_size
        
        for key, value in res.items():
            metrics.log_scalar(
                key, value, sample_size, round=3
            )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
