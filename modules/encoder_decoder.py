from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel

# Function to clone a module N times
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Scaled dot-product attention function
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # Dimension of key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled attention scores
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Apply mask to scores
    p_attn = F.softmax(scores, dim=-1)  # Softmax to get probabilities
    if dropout is not None:
        p_attn = dropout(p_attn)  # Dropout on attention probabilities
    return torch.matmul(p_attn, value), p_attn  # Return attention output and attention weights

# Function to generate a mask for subsequent positions (used in decoder to prevent attending to future positions)
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Transformer model combining encoder and decoder
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask)

# Encoder containing multiple encoder layers and normalization
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # Pass through all layers
        return self.norm(x)

# Encoder layer with self-attention and feed-forward networks
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)  # Two sublayers: attention and feed-forward
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # Self-attention
        return self.sublayer[1](x, self.feed_forward)  # Feed-forward

# Sublayer connection for applying layer normalization and dropout
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Residual connection with dropout

# Layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  # Normalize with gamma and beta

# Decoder with multiple decoder layers and normalization
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask)  # Pass through all layers
        return self.norm(x)

# Decoder layer with self-attention, source-attention, and feed-forward networks
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)  # Three sublayers: self-attn, source-attn, feed-forward

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # Self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # Source-attention
        return self.sublayer[2](x, self.feed_forward)  # Feed-forward

# Multi-headed attention mechanism with dropout
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Ensure d_model is divisible by h
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add extra dimension for batch processing
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # Linear projections and reshape

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # Apply attention

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # Reshape back
        return self.linears[-1](x)  # Final linear projection

# Position-wise feed-forward network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # Apply feed-forward

# Embeddings layer for token embedding lookup
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # Scale embeddings by sqrt(d_model)

# Positional encoding layer to inject position information into embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # Precompute position encodings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))  # Scaling factor for sin/cos
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions: cosine
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Add positional encodings to the input
        return self.dropout(x)

# EncoderDecoder class with model creation and forward pass
class EncoderDecoder(AttModel):
    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            lambda x: x, nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)))
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # Xavier initialization for weights
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)  # Linear projection to vocabulary size

    def init_hidden(self, bsz):
        return []  # Placeholder for hidden states

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # Prepares the features for the encoder and decoder
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, att_masks, seq, seq_mask=None):
        return self.model(fc_feats, seq, att_masks, seq_mask)  # Forward pass

