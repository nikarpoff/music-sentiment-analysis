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
import torch.nn as nn


class SmallSpectrogramTransformer(nn.Module):
    def __init__(self, output_dim: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_UNITS = [128, 256, 512, 1024, 512]
        CNN_KERNELS = [3] * len(CNN_UNITS)
        CNN_STRIDES = [2] * len(CNN_UNITS)
        CNN_PADDINGS = [0] * len(CNN_UNITS)
        CNN_RES_CON = [False] * len(CNN_UNITS)
        RNN_UNITS = 256
        RNN_LAYERS = 2
        TRANSFORMER_DEPTH = 312
        NHEAD = 6
        NUM_ENCODERS = 4

        TRANSFORMER_DROPOUT = 0.3

        self.congruformer = ConGRUFormer(
            in_channels=96,
            cnn_units=CNN_UNITS,
            cnn_kernel_sizes=CNN_KERNELS,
            cnn_strides=CNN_STRIDES,
            cnn_paddings=CNN_PADDINGS,
            cnn_res_con=CNN_RES_CON,
            rnn_units=RNN_UNITS,
            rnn_layers=RNN_LAYERS,
            transformer_depth=TRANSFORMER_DEPTH,
            nhead=NHEAD,
            num_encoders=NUM_ENCODERS,
            dropout=dropout,
            transformer_dropout=TRANSFORMER_DROPOUT,
            device=device
        )

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

    def forward(self, x):
        # CNN Feature extraction
        x = self.congruformer(x)  # (batch, cnn_units, seq)
        return self.output_proj(x)

class SpectrogramTransformer(nn.Module):
    def __init__(self, output_dim: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device
        
        # Model params
        CNN_UNITS = [512, 512, 1024, 1024, 512]
        CNN_KERNELS = [3] * len(CNN_UNITS)
        CNN_STRIDES = [2] * len(CNN_UNITS)
        CNN_PADDINGS = [1] * len(CNN_UNITS)
        CNN_RES_CON = [False] * len(CNN_UNITS)
        RNN_UNITS = 256
        RNN_LAYERS = 2
        TRANSFORMER_DEPTH = 432
        NHEAD = 6
        NUM_ENCODERS = 4

        TRANSFORMER_DROPOUT = 0.3

        self.congruformer = ConGRUFormer(
            in_channels=96,
            cnn_units=CNN_UNITS,
            cnn_kernel_sizes=CNN_KERNELS,
            cnn_strides=CNN_STRIDES,
            cnn_paddings=CNN_PADDINGS,
            cnn_res_con=CNN_RES_CON,
            rnn_units=RNN_UNITS,
            rnn_layers=RNN_LAYERS,
            transformer_depth=TRANSFORMER_DEPTH,
            nhead=NHEAD,
            num_encoders=NUM_ENCODERS,
            dropout=dropout,
            transformer_dropout=TRANSFORMER_DROPOUT,
            device=device
        )

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

    def forward(self, x):
        # CNN Feature extraction
        x = self.congruformer(x)  # (batch, cnn_units, seq)
        return self.output_proj(x)
    

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


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom transformer encoder layer with oveloaded forward:
        this encoder applyes Multi-Head Attention with external query q.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, q):
        if self.norm_first:
            x = self.norm1(x)

        x, _ = self.self_attn(query=q,
                              key=x,
                              value=x)
        
        out = q + self.dropout1(x)

        if not self.norm_first:
            out = self.norm1(out)
        else:
            out = self.norm2(out)
        
        x = self.linear2(self.dropout(self.activation(self.linear1(out))))

        out = out + self.dropout2(x)

        if not self.norm_first:
            out = self.norm2(out)
        
        return out

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer
            for _ in range(num_layers)
        ])

    def forward(self, src, q):
        output = q
        for layer in self.layers:
            output = layer(src, output)
        return output


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


class MultiLayerConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            cnn_units: list,
            cnn_kernel_sizes: list,
            cnn_strides: list,
            cnn_paddings: list,
            residual_connections: list,
            dropout=0.1,
            device='cuda'
    ) -> None:
        """
        Multi-layer 1D convolution.
        """
        super().__init__()
        assert len(cnn_units) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_paddings) == len(residual_connections)
        assert len(cnn_units) != 0
        self.device = device

        conv_layers = []
        current_in = in_channels

        for out_channels, kernel_size, stride, padding in zip(cnn_units, cnn_kernel_sizes, cnn_strides, cnn_paddings):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
                
            current_in = out_channels
        
        self.model = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConGRUFormer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            cnn_units: list,
            cnn_kernel_sizes: list,
            cnn_strides: list,
            cnn_paddings: list,
            cnn_res_con: list,
            rnn_units=256,
            rnn_layers=1,
            transformer_depth=256,
            nhead=8,
            num_encoders=6,
            dropout=0.2,
            transformer_dropout=0.3,
            device='cuda'
    ) -> None:
        """
        Architecture based on 1D convolution, GRU time-sequence processing and Transformer processing.
        """
        assert rnn_layers != 0

        super().__init__()
        self.device = device
        self.depth = transformer_depth

        # Time axis compression by CNN
        self.cnn = MultiLayerConv1d(
            in_channels,
            cnn_units=cnn_units,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_paddings=cnn_paddings,
            residual_connections=cnn_res_con,
            dropout=dropout,
            device=device
        )
        # Output of CNN is (batch, cnn_units[-1], new_sequence_length)
        depth_cnn = cnn_units[-1]

        self.rnn = nn.GRU(depth_cnn, rnn_units, num_layers=rnn_layers, batch_first=True,
                        dropout=dropout, bidirectional=True)
        depth_rnn = rnn_units * 2
            
        # Linear projection with layer normalization from CNN/RNN output to TRANSFORMER input
        self.x_proj = nn.Sequential(
            nn.Linear(depth_cnn, transformer_depth),
            nn.ReLU()
        )

        self.q_proj = nn.Sequential(
            nn.Linear(depth_rnn, transformer_depth),
            nn.LayerNorm(transformer_depth),
            nn.ReLU()
        )

        # Use sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(transformer_depth)

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=transformer_depth,
            dim_feedforward=transformer_depth * 4,
            nhead=nhead,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True     # LayerNorm first
        )
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.attention_pool = MultiHeadPool(transformer_depth, nhead, transformer_dropout)

    def forward(self, x):
        # CNN Feature extraction
        x = self.cnn(x)  # (batch, cnn_units, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
            
        # RNN processing
        q, _ = self.rnn(x)  # (batch, seq, rnn_units*2)
        
        # Project to transformer dimensions
        x = self.x_proj(x)  # (batch, seq, d_model)
        q = self.q_proj(q)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        q = self.pos_encoder(q)
        
        # Transformer processing
        x = self.transformer_encoder(x, q)  # (batch, seq, d_model)

        # Multi-Head Attention pooling
        x = self.attention_pool(x)  # (batch, d_model)

        return x
