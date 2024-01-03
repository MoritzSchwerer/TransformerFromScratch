#!/usr/bin/env python3

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dims = embed_size // num_heads
        self.keys = nn.Linear(self.head_dims, self.head_dims)
        self.queries = nn.Linear(self.head_dims, self.head_dims)
        self.values = nn.Linear(self.head_dims, self.head_dims)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, keys, queries, values, mask=None):
        N = values.shape[0]
        key_len, query_len, value_len = keys.shape[1], queries.shape[1], values.shape[1]

        keys = keys.reshape(N, key_len, self.num_heads, self.head_dims)
        queries = keys.reshape(N, query_len, self.num_heads, self.head_dims)
        values = keys.reshape(N, value_len, self.num_heads, self.head_dims)

        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        out = out.reshape(N, value_len, self.embed_size)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )

    def forward(self, keys, queries, values, mask=None):
        attention = self.attention(keys, queries, values, mask)
        x = self.dropout(self.norm1(attention + values))
        fwd = self.feed_forward(x)
        out = self.dropout(self.norm2(fwd + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, embed_size: int, num_layers: int,
                 num_heads: int, forward_expansion: int, dropout: float,
                 max_length: int, device='cpu'):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # this is not like in the paper
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[0:2]
        pos = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.pos_embedding(pos)
        )
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out
