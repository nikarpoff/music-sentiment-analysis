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
        CNN_OUT_CHANNELS = 512
        RNN_UNITS = 256
        TRANSFORMER_DEPTH = 256
        NHEAD = 16
        NUM_ENCODERS = 6

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
            dropout=dropout,
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


class SpectrogramMaskedAutoEncoder(nn.Module):
    def __init__(self, mask_ratio=0.8, dropout=0.2, device='cuda'):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.device = device

        # Model params
        INPUT_CHANNELS = 96
        CNN_OUT_CHANNELS = 512
        KERNEL_SIZE = 3
        PADDING = 1
        RNN_UNITS = 256
        TRANSFORMER_DEPTH = 256
        NHEAD = 16
        NUM_ENCODER_LAYERS = 6
        NUM_DECODER_LAYERS = 4
        BIDIRECTIONAL_RNN = True

        # --- Encoder ---
        self.encoder = SpectrogramMaskedEncoder(INPUT_CHANNELS, CNN_OUT_CHANNELS, KERNEL_SIZE, PADDING, RNN_UNITS, TRANSFORMER_DEPTH, NHEAD,
                                                NUM_ENCODER_LAYERS, bidirectional_rnn=BIDIRECTIONAL_RNN, mask_ratio=mask_ratio,
                                                dropout=dropout, device=device)

        # --- Decoder ---
        self.decoder = SpectrogramMaskedDecoder(INPUT_CHANNELS, CNN_OUT_CHANNELS, KERNEL_SIZE, PADDING, RNN_UNITS, TRANSFORMER_DEPTH,
                                                NHEAD, NUM_DECODER_LAYERS, BIDIRECTIONAL_RNN, dropout=dropout, device=device)

    def forward(self, spec):
        """
        spec: Tensor (batch, features, sequence)
        returns: recon_spec (batch, features, sequence)
        """
        initial_len = spec.size(-1)
        encoded_vis_spec, masked_full_spec, mask = self.encoder(spec)       # remember mask also
        decoded_spec = self.decoder(masked_full_spec, encoded_vis_spec)     # decoded spec length may differ

        return nn.functional.interpolate(decoded_spec, size=initial_len, mode='linear')
    

class SpectrogramMaskedEncoder(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, kernel_size, padding, rnn_units, transformer_depth, nhead,
                 num_encoder_layers, mask_ratio=0.8, bidirectional_rnn=True, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device
        self.transformer_depth = transformer_depth
        self.mask_ratio = mask_ratio

        # Encoder with 1D convolution
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size),

            nn.Conv1d(128, 256, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size),

            nn.Conv1d(256, 512, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size),

            nn.Conv1d(512, 512, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size),

            nn.Conv1d(512, cnn_out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size),
        )
        # Output of CNN is (batch, CNN_OUT_CHANNELS, new_sequence_length)

        self.rnn_encoder = nn.GRU(cnn_out_channels, rnn_units, num_layers=2, batch_first=True,
                                  dropout=dropout, bidirectional=bidirectional_rnn)

        # Linear projection with layer normalization from RNN output to TRANSFORMER input
        self.d_model_proj = nn.Sequential(
            nn.Linear(rnn_units * 2, transformer_depth),
            nn.LayerNorm(transformer_depth),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # Use sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(transformer_depth)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_depth,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Trainable mask token (to avoid masking with padding value)
        self.mask_token = nn.Parameter(torch.zeros(transformer_depth))

    def forward(self, x, gen_mask=True):
        # CNN specs encoding. Seq len decreasing
        x = self.cnn_encoder(x)  # (batch, cnn_units, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
        
        # RNN processing
        x, _ = self.rnn_encoder(x)  # (batch, seq, rnn_units*2)
        
        # Project to transformer dimensions
        x = self.d_model_proj(x)  # (batch, seq, d_model)
        
        # Generate random mask if required.
        if gen_mask:
            seq_len = x.size(1)

            # Generate mask (single mask for all of batches)
            num_vis = int(round(seq_len * (1 - self.mask_ratio)))
        
            perm = torch.randperm(seq_len, device=self.device)
            vis_idx  = perm[:num_vis].argsort(dim=0)

            mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            mask[vis_idx] = True

            # Apply mask for BxS along whole depth. Get unmasked and masked X.
            x_vis = x[:, vis_idx, :]  # encode only unmasked parts.

            x_masked = torch.empty_like(x)
            i_x_vis = 0
            for i in range(len(mask)):
                if mask[i]:
                    x_masked[:, i, :] = x_vis[:, i_x_vis, :]
                    i_x_vis += 1
                else:
                    x_masked[:, i, :] = self.mask_token
        else:
            # This case intended for encoder-only architecture (pre-learned, not autoencoder)
            x_vis = x  # if mask not required, encode whole spectrogram.
            x_masked = None
            mask = None

        # Add positional encoding
        x_enc = self.pos_encoder(x_vis)
        
        # Transformer processing
        x_enc = self.transformer_encoder(x_enc)  # (batch, vis_seq, d_model)

        return x_enc, x_masked, mask

    def __str__(self):
        model_describe = ""
        model_describe += "CNN Encoder: " + str(self.cnn_encoder) + "\n" * 2
        model_describe += "RNN Encoder: " + str(self.rnn_encoder) + "\n" * 2
        model_describe += "Projection to transformer depth" + str(self.d_model_proj) + "\n" * 2
        model_describe += str(self.pos_encoder) + "\n" * 2
        model_describe += "Transformer Encoder: " + str(self.transformer_encoder) + "\n"
        return model_describe


class SpectrogramMaskedDecoder(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, kernel_size, padding, rnn_units, transformer_depth,
                 nhead, num_decoder_layers, bidirectional_rnn=True, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_depth,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.rnn_decoder = nn.GRU(transformer_depth, rnn_units, num_layers=2, batch_first=True,
                                  dropout=dropout, bidirectional=bidirectional_rnn)

        # Linear projection with layer normalization from RNN output to CNN input
        self.cnn_units_proj = nn.Sequential(
            nn.Linear(rnn_units * 2, cnn_out_channels),
            nn.LayerNorm(cnn_out_channels),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # Output of CNN decoder is (batch, IN_CHANNELS, IN_SEQ_LEN)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=cnn_out_channels,
                               out_channels=512,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=512,
                               out_channels=512,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=512,
                               out_channels=256,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.ConvTranspose1d(in_channels=128,
                               out_channels=input_channels,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(input_channels),
            nn.GELU(),
        )

    def forward(self, dec_in, enc_out):
        # X(dec_in) - decoder masked input (batch, seq_len, d_model);
        # Memory(enc_out) - encoder output (batch, vis_len, d_model).
        x = self.transformer_decoder(dec_in, enc_out)  # (batch, seq, d_model)

        # RNN decoding
        x, _ = self.rnn_decoder(x)  # (batch, seq, rnn_units)

        # Projection from rnn_units to cnn_units.
        x = self.cnn_units_proj(x)  # (batch, seq, cnn_units)
        x = x.permute(0, 2, 1)  # (batch, cnn_units, seq)

        # Final decoding. Increasing of the seq len.
        x = self.cnn_decoder(x)
        
        return x

    def __str__(self):
        model_describe = ""
        model_describe += "Transformer Decoder: " + str(self.transformer_decoder) + "\n"
        model_describe += "RNN Decoder: " + str(self.rnn_decoder) + "\n" * 2
        model_describe += "Projection to CNN depth" + str(self.cnn_units_proj) + "\n" * 2
        model_describe += "CNN Decoder: " + str(self.cnn_decoder) + "\n" * 2
        return model_describe


class SpectrogramPreTrainedTransformer(nn.Module):
    def __init__(self, encoder: SpectrogramMaskedEncoder | None, output_dim: int, device='cuda'):
        """
        Pre-trained transformer model: encoder -> classifier.
        :param encoder: part of the trained autoencoder. If None then load state dict required.
        :param output_dim: classes to classify number
        """
        super().__init__()
        self.device = device

        self.encoder = encoder
        depth = encoder.transformer_depth

        self.attention_pool = nn.Sequential(
            nn.Linear(depth, depth//2),
            nn.Tanh(),
            nn.Linear(depth//2, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(depth, depth),
            nn.GELU(),
            nn.Linear(depth, depth),
            nn.GELU(),
            nn.Linear(depth, output_dim),
        )

    def forward(self, x):
        x, _, _ = self.encoder(x, gen_mask=False)

        # Attention pooling
        attn_weights = self.attention_pool(x)  # (batch, seq, 1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch, depth)
        
        # Final projection (classification)
        logits = self.output_proj(x)
        return logits

    def __str__(self):
        model_describe = ""
        model_describe += "Encoder: " + str(self.encoder) + "\n"
        model_describe += "Attention pooling: " + str(self.attention_pool) + "\n" * 2
        model_describe += "Output projection: " + str(self.output_proj) + "\n" * 2
        return model_describe

