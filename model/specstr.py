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


import torch
from torch import nn
import math


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


class SpectrogramPureTransformer(nn.Module):
    def __init__(self, output_dim: int, device='cuda'):
        super().__init__()
        self.device = device
        D_MODEL = 96
        NHEAD = 16
        NUM_LAYERS = 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.attention_pool = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2),
            nn.Tanh(),
            nn.Linear(D_MODEL//2, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Linear(D_MODEL, output_dim)

    def forward(self, x):
        # x = truncate_spec(x, self.seq_len)  # shape (batch, mel_features, seq_len)
        x = x.permute(0, 2, 1)                 # shape (batch, seq_len, mel_features)

        x = self.transformer_encoder(x)

        # Attention pooling
        attn_weights = self.attention_pool(x)  # (batch, seq, 1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)

        # Final projection
        logits = self.output_proj(x)
        return logits

    def __str__(self):
        model_describe = ""
        model_describe += str(self.transformer_encoder) + "\n"
        model_describe += str(self.attention_pool) + "\n"
        model_describe += str(self.output_proj) + "\n"
        return model_describe


class SpectrogramTransformer(nn.Module):
    def __init__(self, output_dim: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_OUT_CHANNELS = 1024
        RNN_UNITS = 256
        TRANSFORMER_DEPTH = 256
        NHEAD = 16
        NUM_ENCODERS = 6

        # Mel-spectrograms have size 96x(sequence_size). So in_channels=96
        self.cnn = nn.Sequential(
            ResidualConv1D(in_channels=96, out_channels=128, dropout=0.2),
            nn.MaxPool1d(kernel_size=4, stride=2),

            ResidualConv1D(in_channels=128, out_channels=256, dropout=0.2),
            nn.MaxPool1d(kernel_size=2),

            ResidualConv1D(in_channels=256, out_channels=512, dropout=0.2),
            nn.MaxPool1d(kernel_size=2),

            ResidualConv1D(in_channels=512, out_channels=1024, dropout=0.2),
            nn.MaxPool1d(kernel_size=2),

            ResidualConv1D(in_channels=1024, out_channels=CNN_OUT_CHANNELS, dropout=0.2),
            nn.MaxPool1d(kernel_size=2),
        )
        # Output of CNN is (batch, d_model, new_sequence_length)

        self.rnn = nn.GRU(CNN_OUT_CHANNELS, RNN_UNITS, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=True)

        # Linear projection with layer normalization from RNN output to TRANSFORMER input
        self.d_model_proj = nn.Sequential(
            nn.Linear(RNN_UNITS * 2, TRANSFORMER_DEPTH),
            nn.LayerNorm(TRANSFORMER_DEPTH),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # Use sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(TRANSFORMER_DEPTH)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DEPTH,
            nhead=NHEAD,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODERS)

        self.attention_pool = nn.Sequential(
            nn.Linear(TRANSFORMER_DEPTH, TRANSFORMER_DEPTH//2),
            nn.Tanh(),
            nn.Linear(TRANSFORMER_DEPTH//2, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

    def forward(self, x):
        # CNN Feature extraction
        x = self.cnn(x)  # (batch, cnn_units, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
        
        # RNN processing
        x, _ = self.rnn(x)  # (batch, seq, rnn_units*2)
        
        # Project to transformer dimensions
        x = self.d_model_proj(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)  # (batch, seq, d_model)
        
        # Attention pooling
        attn_weights = self.attention_pool(x)  # (batch, seq, 1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)
        
        # Final projection
        logits = self.output_proj(x)
        return logits

    def __str__(self):
        model_describe = ""
        model_describe += "CNN: " + str(self.cnn) + "\n" * 2
        model_describe += "RNN: " + str(self.rnn) + "\n" * 2
        model_describe += "Projection to transformer depth" + str(self.d_model_proj) + "\n" * 2
        model_describe += str(self.pos_encoder) + "\n" * 2
        model_describe += "Transformer Encoder: " + str(self.transformer_encoder) + "\n"
        model_describe += "Attention pooling: " + str(self.attention_pool) + "\n" * 2
        model_describe += "Output projection: " + str(self.output_proj) + "\n" * 2
        return model_describe
