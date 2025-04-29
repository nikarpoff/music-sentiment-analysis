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

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x
    

class SpectrogramPureTransformer(nn.Module):
    def __init__(self, d_model: int, output_dim: int, nhead: int, num_layers: int, seq_len: int, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.linear = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x):
        # x = truncate_spec(x, self.seq_len)  # shape (batch, mel_features, seq_len)
        x = x.permute(0, 2, 1)                 # shape (batch, seq_len, mel_features)

        x = self.transformer_encoder(x)
        logits = self.linear(x[:, -1, :])

        return logits

    def __str__(self):
        model_describe = ""
        model_describe += str(self.transformer_encoder) + "\n"
        model_describe += str(self.linear) + "\n"
        return model_describe

class SpectrogramTransformer(nn.Module):
    def __init__(self, cnn_units: int, rnn_units: int, d_model: int, output_dim: int, nhead: int, num_layers: int,
                 seq_len: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.device = device
        self.cnn_units = cnn_units
        self.rnn_units = rnn_units
        self.dropout = dropout

        # Mel-spectrograms have size 96x(sequence_size). So in_channels=96
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=cnn_units, kernel_size=3, padding=2),
            nn.BatchNorm1d(cnn_units),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Output of CNN is (batch, d_model, new_sequence_length)

        self.rnn = nn.GRU(cnn_units, rnn_units, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=True)

        self.d_model_proj = nn.Sequential(
            nn.Linear(rnn_units * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Linear(self.d_model, self.output_dim)

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
        model_describe += str(self.cnn) + "\n" * 2
        model_describe += str(self.rnn) + "\n" * 2
        model_describe += str(self.d_model_proj) + "\n" * 2
        model_describe += str(self.transformer_encoder) + "\n"
        model_describe += str(self.output_proj) + "\n" * 2
        return model_describe
