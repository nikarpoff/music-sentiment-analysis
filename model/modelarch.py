import torch
import math
from torch import nn


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


class SpectrogramTransformer9M(nn.Module):
    def __init__(self, output_dim: int, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_OUT_CHANNELS = 512
        RNN_UNITS = 256
        TRANSFORMER_DEPTH = 256
        NHEAD = 16
        NUM_ENCODERS = 6

        TRANSFORMER_DROPOUT = 0.
        dropout = 0.

        # Mel-spectrograms have size 96x(sequence_size). So in_channels=96
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=512, out_channels=CNN_OUT_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(CNN_OUT_CHANNELS),
            nn.GELU(),
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
            dim_feedforward=TRANSFORMER_DEPTH * 4,
            nhead=NHEAD,
            dropout=TRANSFORMER_DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True     # LayerNorm first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODERS, enable_nested_tensor=False)

        self.attention_pool = nn.Sequential(
            nn.Linear(TRANSFORMER_DEPTH, TRANSFORMER_DEPTH//2),
            nn.Tanh(),
            nn.Linear(TRANSFORMER_DEPTH//2, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.d_model_proj(x)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        attn_weights = self.attention_pool(x)
        x = torch.sum(x * attn_weights, dim=1)
        
        logits = self.output_proj(x)
        return logits