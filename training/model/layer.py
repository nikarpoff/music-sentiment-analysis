# Copyright [2024] [Nikita Karpov]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch
from torch import nn

class MultiHeadPool(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.3):
        """
        Multi-head Attention Pool.
        """
        super().__init__()
        
        self.query = nn.Parameter(torch.randn(1, d_model))  # learnable query
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, depth)
        
        # out: (batch, 1, depth), attn_weights: (batch, 1, seq_len)
        attn_out, attn_weights = self.mha(q, x, x, need_weights=True)  # K, V = x, x
        out = attn_out.squeeze(1)  # (B, D)

        # Residual connection & normalization
        out = self.norm(out + q.squeeze(1))
        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # positions from 0 to max_len-1

        # sinusoidal absolute, wk = 1 / 1000 ^ (2k / d)
        omega = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * omega)  # even code with sin
        pe[:, 1::2] = torch.cos(position * omega)  # odd code with cos

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x
    
    def __str__(self):
        return "Positional sinusoidal encoding"


class ResidualConv1D(nn.Module):
    """
    ResNet-like architecture with two Conv1D and residual connection.
    Kernel size, padding and stride are chosen in such a way that there is no reduction in the seq_len.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.act1  = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        
        # If in_channels != out_channels then apply 1x1 convolution (for identity=x).
        self.identity_projection = None
        if in_channels != out_channels:
            self.identity_projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act2 = nn.GELU()
    
    def forward(self, x):
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Projection (if required)
        if self.identity_projection is not None:
            identity = self.identity_projection(identity)
        
        # Residual connection and out activation
        out = out + identity
        out = self.act2(out)
        out = self.dropout(out)

        return out


class MultiScaleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=k//2)
            for k in (3, 7, 15, 31)
        ])

        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.GELU()

    def forward(self, x):
        outs = [conv(x) for conv in self.branches]
        x = torch.cat(outs, dim=1)
        
        return self.act(self.bn(x))
