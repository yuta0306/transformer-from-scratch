from typing import Optional

import torch
import torch.nn as nn

from transformer.decoder import TransformerDecoder, TransformerDecoderLayer
from transformer.embedding import PositionalEncoding
from transformer.encoder import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        tgt_vocab_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        max_length: int = 512,
    ) -> None:
        super(Transformer, self).__init__()

        if hidden_size is None:
            hidden_size = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = None
        if tgt_vocab_size is not None:
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pe = PositionalEncoding(d_model, dropout, max_length=max_length)
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, hidden_size, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, hidden_size, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = self.embedding(src)
        src = self.pe(src)

        context = self.encoder(src, src_mask)

        if self.tgt_embedding is not None:
            tgt = self.tgt_embedding(tgt)
        else:
            tgt = self.embedding(tgt)
        tgt = self.pe(tgt)

        out = self.decoder(tgt, context, tgt_mask, context_mask)

        return out

    @staticmethod
    def generate_square_subsequent_mask(size: int) -> torch.Tensor:
        return torch.tril(torch.full((size, size), 1), diagonal=0)
