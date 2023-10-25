import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
import numpy as np
import math
from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_weights = None
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # A = self.dropout(torch.softmax(scores, dim=-1))
        self.attention_weights = A.detach().cpu().numpy()
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, num_features, n_heads, d_keys=16, d_values=16, mix=True, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(num_features, d_keys * n_heads)
        self.key_projection = nn.Linear(num_features, d_keys * n_heads)
        self.value_projection = nn.Linear(num_features, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, num_features)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # print(queries.shape, keys.shape,values.shape)

        out = self.inner_attention(queries,keys,values)

        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)

class TimeAttentionLayer(nn.Module):
    def __init__(self, num_features, time_steps, n_heads, d_ff=None, dropout=0.1):
        super(TimeAttentionLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        
        # Time attention on sequence length
        self.time_attention = AttentionLayer(num_features, n_heads, dropout=dropout)
        
        # Dimension attention on features
        # self.dim_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        # self.norm3 = nn.LayerNorm(d_model)
        
        self.MLP1 = nn.Sequential(nn.Linear(num_features, d_ff),nn.GELU(),nn.Linear(d_ff, num_features))
        
        # self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),nn.GELU(),nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Time Attention Stage
        B, L, _ = x.shape

        # print('in time attention', x.shape)
        time_enc = self.time_attention(x, x, x)
        time_out = x + self.dropout(time_enc)
        time_out = self.norm1(time_out)
        time_out = time_out + self.dropout(self.MLP1(time_out))
        time_out = self.norm2(time_out)
        # print('out time attention', time_out.shape)
        
        # # Rearrange for Dimension Attention Stage
        # dim_in = rearrange(time_out, 'b l d -> b d l')
        # dim_enc = self.dim_attention(dim_in, dim_in, dim_in)
        # dim_out = dim_in + self.dropout(dim_enc)
        # dim_out = self.norm3(dim_out)
        
        # # Rearrange back to original format
        # final_out = rearrange(dim_out, 'b d l -> b l d')

        return time_out


class FeatAttentionLayer(nn.Module):### exactly same as time attention
    def __init__(self, num_features, time_steps, n_heads, d_ff=None, dropout=0.1):
        super(FeatAttentionLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        
        # Time attention on sequence length
        # self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        
        # Dimension attention on features
        self.dim_attention = AttentionLayer(num_features, n_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(num_features)
        self.norm4 = nn.LayerNorm(num_features)
        # self.norm3 = nn.LayerNorm(d_model)
        
        # self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),nn.GELU(),nn.Linear(d_ff, d_model))
        
        self.MLP2 = nn.Sequential(nn.Linear(num_features, d_ff),nn.GELU(),nn.Linear(d_ff, num_features))

    def forward(self, x):
        # Time Attention Stage
        B, L, _ = x.shape

        # print('in feat attention', x.shape)
        dim_enc = self.dim_attention(x, x, x)
        dim_out = x + self.dropout(dim_enc)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out))
        dim_out = self.norm4(dim_out)
        # print('out time attention', dim_out.shape)
        
        # # Rearrange for Dimension Attention Stage
        # dim_in = rearrange(time_out, 'b l d -> b d l')
        # dim_enc = self.dim_attention(dim_in, dim_in, dim_in)
        # dim_out = dim_in + self.dropout(dim_enc)
        # dim_out = self.norm3(dim_out)
        
        # # Rearrange back to original format
        # final_out = rearrange(dim_out, 'b d l -> b l d')

        return dim_out


class TimeSeriesClassifier(nn.Module):
    def __init__(self, num_features, num_classes, time_steps, n_heads, d_ff=32, dropout=0.5):
        super(TimeSeriesClassifier, self).__init__()
        
        # self.time_projection = nn.Linear(num_features, d_model)
        self.time_attention = TimeAttentionLayer(num_features, time_steps, n_heads, d_ff, dropout)

        # self.feat_projection = nn.Linear(time_steps, d_model)
        self.feat_attention = FeatAttentionLayer(time_steps, num_features, n_heads, d_ff, dropout)


        self.classifier = nn.Linear(2*time_steps*num_features, num_classes)


    def forward(self, x):

        # inp_time = self.time_projection(x)
        inp_time = x
        inp_time = self.time_attention(inp_time)
        inp_time = inp_time.view(inp_time.size(0), -1)

        # inp_feat = self.feat_projection(x.permute(0, 2, 1))
        inp_feat = x.permute(0, 2, 1)
        inp_feat = self.feat_attention(inp_feat)
        inp_feat = inp_feat.view(inp_feat.size(0), -1)

        # print(inp_time.shape, inp_feat.shape)
        out = torch.cat((inp_time, inp_feat), dim=1)
        # x = self.classifier(x)
        out = self.classifier(out)
        return out
