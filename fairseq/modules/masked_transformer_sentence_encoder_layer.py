# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.masked_modules import MaskedLayerNorm_select, MaskedLinear
from fairseq.modules import MultiheadAttention
from fairseq.modules.masked_multihead_attention import MaskedMultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class MaskedTransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        args,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            args,
            self.embedding_dim,
            num_attention_heads,
            attention_dropout,
            True,
            q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = MaskedLayerNorm_select(self.embedding_dim, args.mask_layernorm_type)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            args,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise,
            qn_block_size,
            args,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = MaskedLayerNorm_select(self.embedding_dim, args.mask_layernorm_type)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size, args):
        return quant_noise(
            MaskedLinear(input_dim, output_dim, bias=args.need_bias, prune_ratio=args.prune_ratio, 
                prune_method=args.prune_method, mask_init=args.mask_init, mask_constant=args.mask_constant,
                init=args.init, nonlinearity=args.activation_fn, scale_fan=args.scale_fan), 
                p=q_noise, block_size=qn_block_size
        )


    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size, args):
        return quant_noise(
            MaskedLinear(input_dim, output_dim, bias=args.need_bias, prune_ratio=args.prune_ratio, 
                prune_method=args.prune_method, mask_init=args.mask_init, mask_constant=args.mask_constant,
                init=args.init, nonlinearity=args.activation_fn, scale_fan=args.scale_fan), 
                p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(
        self,
        args,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MaskedMultiheadAttention(
            args,
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        # return MultiheadAttention(
        #     embed_dim,
        #     num_attention_heads,
        #     dropout=dropout,
        #     self_attention=True,
        #     q_noise=q_noise,
        #     qn_block_size=qn_block_size,
        # )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
