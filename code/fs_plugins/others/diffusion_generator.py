# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import utils
from functools import partial


DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)


def denoised_fn_round(model, text_emb):
    # thresh_t = 50
    # # print(thresh_t)
    # if thresh_t is not None and t[0] > thresh_t:
    #     return text_emb

    # if args.model_arch == '1d-unet':
    #     text_emb = text_emb.permute(0, 2, 1)
    # return text_emb
    # print(t.float().mean(), t[0])

    # assert t.float().mean() == t[0].float()
    
    # print(text_emb.shape) # bsz, seqlen, dim
    down_proj_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb**2).sum(-1).view(-1, 1) #vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) #d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) #bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t) #(vocab, d) x (d, bsz*seqlen)
            dist = torch.clamp(dist, 0.0, np.inf)
            # print(dist.shape)
        topk_out = torch.topk(-dist, k=1, dim=0)
        #     adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
        #         down_proj_emb.size(0), -1, -1)
        #     adjacency = -th.norm(adjacency, dim=-1)
        # topk_out = th.topk(adjacency, k=1, dim=0)
        # print(topk_out1.indices == topk_out.indices)
        # assert th.all(topk_out1.indices == topk_out.indices)
        return topk_out.values, topk_out.indices

    def get_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                down_proj_emb.size(0), -1, -1)
            adjacency = - torch.norm(adjacency, dim=-1)
        topk_out = torch.topk(adjacency, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(down_proj_emb,
    #                        text_emb.to(down_proj_emb.device), dist=dist)
    val, indices = get_efficient_knn(down_proj_emb,
                           text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    
    return new_embeds


class DiffusionGenerator(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
        diffusion_module=None,
        diff_steps=None,
        schedule_sampler=None
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.beam_size = beam_size
        self.reranking = reranking
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.models = models
        
        # For diffusion
        self.diffusion = diffusion_module
        self.diff_steps = diff_steps
        self.schedule_sampler = schedule_sampler        
        

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])
        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)

        if self.beam_size > 1:
            assert (
                model.allow_length_beam
            ), "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, length_beam_order
            )
            prev_decoder_out = model.regenerate_length_beam(
                prev_decoder_out, self.beam_size
            )
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])

        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }

        # NOTE: Just one step
        step = 0
        decoder_options = {
            "eos_penalty": self.eos_penalty,
            "max_ratio": self.max_ratio,
            "decoding_format": self.decoding_format,
        }
        prev_decoder_out = prev_decoder_out._replace(
            step=step,
            max_step=self.max_iter + 1,
        )

        ## NOTE: Just modify the model here 
        
        # STEP1: Gaussian p sample loop
        
        
        rounding_emb = partial(denoised_fn_round, model.get_embedding_layer())
        
        device = next(model.parameters()).device
        shape = prev_decoder_out[0].shape
        shape = (*shape, encoder_out["encoder_out"][0].shape[-1])
        
        noise = None
        if noise is not None:
            p_x_t = noise
        else:
            p_x_t = torch.randn(*shape, device=device)
        
        indices = list(range(self.diff_steps))[::-1]
        # indices = list(range(10))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.diffusion.p_sample(
                    model,
                    p_x_t,
                    t,
                    clip_denoised=False,
                    denoised_fn=rounding_emb,
                    model_kwargs={
                        "encoder_out": encoder_out,
                        "prev_decoder_out": prev_decoder_out,
                        "decoder_options": decoder_options
                        },
                    top_p=-1,
                )
                # if langevin_func is not None:
                #     out['t'] = t
                #     out['img'] = img 
                #     out = langevin_func(out)
                p_x_t = out["sample"]

        
        output_logits = model.decoder.forward_x_0_logits(p_x_t)
        # output_tokens = output_logits.argmax(-1)
        # output_scores = output_logits.max(-1)
        
        output_scores, output_tokens = output_logits.max(-1)
        
        decoder_out =prev_decoder_out
        decoder_out = decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
        )

        if self.adaptive:
            # terminate if there is a loop
            terminated, out_tokens, out_scores, out_attn = is_a_loop(
                prev_output_tokens,
                decoder_out.output_tokens,
                decoder_out.output_scores,
                decoder_out.attn,
            )
            decoder_out = decoder_out._replace(
                output_tokens=out_tokens,
                output_scores=out_scores,
                attn=out_attn,
            )

        else:
            terminated = decoder_out.output_tokens.new_zeros(
                decoder_out.output_tokens.size(0)
            ).bool()

        if step == self.max_iter:  # reach last iteration, terminate
            terminated.fill_(1)

        terminated.fill_(1) # Note: force terminating
        
        # collect finalized sentences
        finalized_idxs = sent_idxs[terminated]
        finalized_tokens = decoder_out.output_tokens[terminated]
        finalized_scores = decoder_out.output_scores[terminated]
        finalized_attn = (
            None
            if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
            else decoder_out.attn[terminated]
        )

        if self.retain_history:
            finalized_history_tokens = [h[terminated] for h in decoder_out.history]

        for i in range(finalized_idxs.size(0)):
            finalized[finalized_idxs[i]] = [
                finalized_hypos(
                    step,
                    finalized_tokens[i],
                    finalized_scores[i],
                    None if finalized_attn is None else finalized_attn[i],
                )
            ]

            if self.retain_history:
                finalized[finalized_idxs[i]][0]["history"] = []
                for j in range(len(finalized_history_tokens)):
                    finalized[finalized_idxs[i]][0]["history"].append(
                        finalized_hypos(
                            step, finalized_history_tokens[j][i], None, None
                        )
                    )

        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )

            # aggregate information from length beam
            finalized = [
                finalized[
                    np.argmax(
                        [
                            finalized[self.beam_size * i + j][0]["score"]
                            for j in range(self.beam_size)
                        ]
                    )
                    + self.beam_size * i
                ]
                for i in range(len(finalized) // self.beam_size)
            ]
        return finalized
    

    def rerank(self, reranker, finalized, encoder_input, beam_size):
        def rebuild_batch(finalized):
            finalized_tokens = [f[0]["tokens"] for f in finalized]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0]
                .new_zeros(len(finalized_tokens), finalized_maxlen)
                .fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        final_output_tokens[
            :, 0
        ] = self.eos  # autoregressive model assumes starting with EOS

        reranker_encoder_out = reranker.encoder(*encoder_input)
        length_beam_order = (
            utils.new_arange(
                final_output_tokens, beam_size, reranker_encoder_out.encoder_out.size(1)
            )
            .t()
            .reshape(-1)
        )
        reranker_encoder_out = reranker.encoder.reorder_encoder_out(
            reranker_encoder_out, length_beam_order
        )
        reranking_scores = reranker.get_normalized_probs(
            reranker.decoder(final_output_tokens[:, :-1], reranker_encoder_out),
            True,
            None,
        )
        reranking_scores = reranking_scores.gather(2, final_output_tokens[:, 1:, None])
        reranking_masks = final_output_tokens[:, 1:].ne(self.pad)
        reranking_scores = (
            reranking_scores[:, :, 0].masked_fill_(~reranking_masks, 0).sum(1)
        )
        reranking_scores = reranking_scores / reranking_masks.sum(1).type_as(
            reranking_scores
        )

        for i in range(len(finalized)):
            finalized[i][0]["score"] = reranking_scores[i]

        return finalized
