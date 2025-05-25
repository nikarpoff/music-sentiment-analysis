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


from torch import nn
from model.layer import *


# class RawAudioTransformer(nn.Module):
#     def __init__(self, output_dim: int, dropout=0.2, device='cuda'):
#         super().__init__()
#         self.device = device

#         # Model params
#         CNN_OUT_CHANNELS = 512
#         RNN_UNITS = 256
#         TRANSFORMER_DEPTH = 256
#         NHEAD = 8
#         NUM_ENCODERS = 4

#         TRANSFORMER_DROPOUT = 0.3

#         # Audio has a lot os samples -> downsampling required -> large CNN with residual connections 
#         # self.cnn = nn.Sequential(
#         #     # Downsampling: (L) / 2 each block
#         #     nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(4),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(8),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(16),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(32),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(64),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(128),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(256),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Conv1d(in_channels=256, out_channels=CNN_OUT_CHANNELS, kernel_size=3, stride=2, bias=False),
#         #     nn.BatchNorm1d(CNN_OUT_CHANNELS),
#         #     nn.GELU(),
#         #     # nn.MaxPool1d(kernel_size=2),

#         #     nn.Dropout(dropout),
#         # )
#         # Total blocks is 8 -> there is downsampling in 256 times.
#         # Output of CNN is (batch, d_model, new_sequence_length)

#         self.cnn = nn.Sequential(
#             MultiScaleConv1d(1, 4),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(dropout),
            
#             MultiScaleConv1d(16, 32),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(dropout),
            
#             MultiScaleConv1d(128, 128),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(dropout),

#             nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2, bias=False),
#             nn.BatchNorm1d(512),
#             nn.GELU(),
#             nn.MaxPool1d(kernel_size=4),
#         )

#         self.rnn = nn.GRU(512, RNN_UNITS, num_layers=4, batch_first=True,
#                           dropout=dropout, bidirectional=True)

#         # Linear projection with layer normalization from RNN output to TRANSFORMER input
#         self.d_model_proj = nn.Sequential(
#             nn.Linear(RNN_UNITS * 2, TRANSFORMER_DEPTH),
#             nn.LayerNorm(TRANSFORMER_DEPTH),
#             nn.Dropout(dropout),
#             nn.ReLU()
#         )

#         # # Use sinusoidal positional encoding
#         # self.pos_encoder = PositionalEncoding(TRANSFORMER_DEPTH)
#         # encoder_layer = nn.TransformerEncoderLayer(
#         #     d_model=TRANSFORMER_DEPTH,
#         #     dim_feedforward=TRANSFORMER_DEPTH * 4,
#         #     nhead=NHEAD,
#         #     dropout=TRANSFORMER_DROPOUT,
#         #     activation='gelu',
#         #     batch_first=True,
#         #     norm_first=True     # LayerNorm first
#         # )
#         # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODERS, enable_nested_tensor=False)

#         self.attention_pool = MultiHeadPool(TRANSFORMER_DEPTH, NHEAD, TRANSFORMER_DROPOUT)

#         self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

#     def forward(self, x):
#         # CNN Feature extraction
#         x = self.cnn(x)  # (batch, cnn_units, seq)
#         x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
        
#         # RNN processing
#         x, _ = self.rnn(x)  # (batch, seq, rnn_units*2)
        
#         # Project to transformer dimensions
#         x = self.d_model_proj(x)  # (batch, seq, d_model)
        
#         # # Add positional encoding
#         # x = self.pos_encoder(x)
        
#         # # Transformer processing
#         # x = self.transformer_encoder(x)  # (batch, seq, d_model)

#         # Multi-Head Attention pooling
#         x = self.attention_pool(x)  # (batch, d_model)

#         # Final projection
#         logits = self.output_proj(x)
#         return logits

#     def __str__(self):
#         model_describe = ""
#         model_describe += "CNN: " + str(self.cnn) + "\n" * 2
#         # # model_describe += "RNN: " + str(self.rnn) + "\n" * 2
#         # model_describe += "Projection to transformer depth" + str(self.d_model_proj) + "\n" * 2
#         # model_describe += str(self.pos_encoder) + "\n" * 2
#         # model_describe += "Transformer Encoder: " + str(self.transformer_encoder) + "\n"
#         # model_describe += "Attention pooling: " + str(self.attention_pool) + "\n" * 2
#         # model_describe += "Output projection: " + str(self.output_proj) + "\n" * 2
#         return model_describe

class RawAudioTransformer(nn.Module):
    def __init__(self, output_dim: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_OUT_CHANNELS = 1024
        RNN_UNITS = 256
        TRANSFORMER_DEPTH = 256
        NHEAD = 8
        NUM_ENCODERS = 6

        TRANSFORMER_DROPOUT = 0.1

        # Audio has a lot os samples -> downsampling required -> large CNN with residual connections 
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=10, stride=5, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Conv1d(in_channels=1024, out_channels=CNN_OUT_CHANNELS, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(CNN_OUT_CHANNELS),
            nn.GELU(),

            nn.Dropout(dropout),
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

        self.attention_pool = MultiHeadPool(TRANSFORMER_DEPTH, NHEAD, TRANSFORMER_DROPOUT)

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_dim)

    def forward(self, x):
        # CNN Feature extraction
        x = self.cnn(x)  # (batch, cnn_units, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
        # print(x.shape)
        
        # RNN processing
        x, _ = self.rnn(x)  # (batch, seq, rnn_units*2)
        
        # Project to transformer dimensions
        x = self.d_model_proj(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)  # (batch, seq, d_model)

        # Multi-Head Attention pooling
        x = self.attention_pool(x)  # (batch, d_model)

        # Final projection
        logits = self.output_proj(x)
        return logits

    def __str__(self):
        model_describe = ""
        model_describe += "CNN: " + str(self.cnn) + "\n" * 2
        # # model_describe += "RNN: " + str(self.rnn) + "\n" * 2
        # model_describe += "Projection to transformer depth" + str(self.d_model_proj) + "\n" * 2
        # model_describe += str(self.pos_encoder) + "\n" * 2
        # model_describe += "Transformer Encoder: " + str(self.transformer_encoder) + "\n"
        # model_describe += "Attention pooling: " + str(self.attention_pool) + "\n" * 2
        # model_describe += "Output projection: " + str(self.output_proj) + "\n" * 2
        return model_describe
