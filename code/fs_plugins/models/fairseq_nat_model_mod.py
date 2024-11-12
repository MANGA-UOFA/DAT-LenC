# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

import torch
from fs_plugins.models.transformer.transformer_encoder import TransformerEncoder
from fs_plugins.models.transformer.transformer_decoder import TransformerDecoder
from fs_plugins.models.transformer.transformer_legacy import TransformerModel

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.models.transformer import Embedding

from fairseq import utils

def ensemble_encoder(func):
    def wrapper(self, *args, **kwargs):
        if self.ensemble_models is None or len(self.ensemble_models) == 1:
            return func(self, *args, **kwargs)
        encoder_outs = [
            func(model, *args, **kwargs, return_all_hiddens=True)
            for model in self.ensemble_models
        ]
        _encoder_out = encoder_outs[0].copy()

        def stack(key):
            outs = [e[key][0] for e in encoder_outs]
            return [torch.stack(outs, -1) if outs[0] is not None else None]

        _encoder_out["encoder_out"] = stack("encoder_out")
        _encoder_out["encoder_embedding"] = stack("encoder_embedding")

        num_layers = len(_encoder_out["encoder_states"])
        if num_layers > 0:
            _encoder_out["encoder_states"] = [
                torch.stack([e["encoder_states"][i] for e in encoder_outs], -1)
                for i in range(num_layers)
            ]
        return _encoder_out

    return wrapper


def ensemble_decoder(func):
    def wrapper(self, normalize=False, encoder_out=None, *args, **kwargs):
        if self.ensemble_models is None or len(self.ensemble_models) == 1:
            return func(
                self, normalize=normalize, encoder_out=encoder_out, *args, **kwargs
            )

        def _replace(encoder_out, new_val):
            new_encoder_out = encoder_out.copy()
            new_encoder_out["encoder_out"] = [new_val]
            return new_encoder_out

        action_outs = [
            func(
                model,
                normalize=normalize,
                encoder_out=_replace(
                    encoder_out, encoder_out["encoder_out"][0][:, :, :, i]
                ),
                *args,
                **kwargs
            )
            for i, model in enumerate(self.ensemble_models)
        ]

        if not isinstance(action_outs[0], tuple):  # return multiple values
            action_outs = [[a] for a in action_outs]
        else:
            action_outs = [list(a) for a in action_outs]

        ensembled_outs = []
        for i in range(len(action_outs[0])):
            if i == 0 and normalize:
                ensembled_outs += [
                    torch.logsumexp(
                        torch.stack([a[i] for a in action_outs], -1), dim=-1
                    )
                    - math.log(len(self.ensemble_models))
                ]
            elif action_outs[0][i] is not None:
                ensembled_outs += [torch.stack([a[i] for a in action_outs], -1)]
            else:
                ensembled_outs += [None]

        if len(ensembled_outs) == 1:
            return ensembled_outs[0]
        return tuple(ensembled_outs)

    return wrapper

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t

class FairseqNATModel(TransformerModel):
    """
    Abstract class for all nonautoregressive-based models
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

        self.ensemble_models = None

    @property
    def allow_length_beam(self):
        return False

    @property
    def allow_ensemble(self):
        return True

    def enable_ensemble(self, models):
        self.encoder.ensemble_models = [m.encoder for m in models]
        self.decoder.ensemble_models = [m.decoder for m in models]

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = FairseqNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = FairseqNATEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(self, *args, **kwargs):
        return NotImplementedError

    def initialize_output_tokens(self, *args, **kwargs):
        return NotImplementedError

    def forward(self, *args, **kwargs):
        return NotImplementedError


class FairseqNATEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.ensemble_models = None

    @ensemble_encoder
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class FairseqNATDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.ensemble_models = None



class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt