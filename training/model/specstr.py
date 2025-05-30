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
from model.layer import *


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
        CNN_UNITS = [128, 256, 512, 1024, 512]
        CNN_KERNELS = [3] * len(CNN_UNITS)
        CNN_STRIDES = [2] * len(CNN_UNITS)
        CNN_PADDINGS = [0] * len(CNN_UNITS)
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
            cnn_res_con=False,
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
    

class SpectrogramMaskedAutoEncoder(nn.Module):
    def __init__(self, mask_ratio=0.8, dropout=0.2, device='cuda'):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.device = device

        # Model params
        INPUT_CHANNELS = 96
        KERNEL_SIZE = 2
        PADDING = 1
        
        TRANSFORMER_DEPTH = 256
        NHEAD = 8
        NUM_ENCODER_LAYERS = 6
        NUM_DECODER_LAYERS = 4
        TRANSFORMER_CONTEXT_LEN = 128

        DROPOUT_CNN = 0.1
        DROPOUT_TRANSFORMER = 0.4

        # --- Encoder ---
        self.encoder = SpectrogramMaskedEncoder(INPUT_CHANNELS, KERNEL_SIZE, PADDING, TRANSFORMER_DEPTH, TRANSFORMER_CONTEXT_LEN,
                                                NHEAD, NUM_ENCODER_LAYERS, mask_ratio=mask_ratio, dropout_cnn=DROPOUT_CNN,
                                                dropout_transformer=DROPOUT_TRANSFORMER, device=device)

        # --- Decoder ---
        self.decoder = SpectrogramMaskedDecoder(INPUT_CHANNELS, KERNEL_SIZE, PADDING, TRANSFORMER_DEPTH, NHEAD, NUM_DECODER_LAYERS,
                                                DROPOUT_CNN, DROPOUT_TRANSFORMER, device=device)

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
    def __init__(self, input_channels, kernel_size, padding, transformer_depth, transformer_context_len,
                 nhead, num_encoder_layers, mask_ratio=0.8, dropout_cnn=0.1, dropout_transformer=0.4, device='cuda'):
        super().__init__()
        self.device = device
        self.nhead = nhead
        self.transformer_depth = transformer_depth
        self.mask_ratio = mask_ratio

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=kernel_size),
            nn.Dropout1d(dropout_cnn),

            # nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding, bias=False),
            # nn.BatchNorm1d(256),
            # nn.GELU(),
            # nn.MaxPool1d(kernel_size=kernel_size),
            # nn.Dropout1d(dropout_cnn),

            nn.Conv1d(in_channels=128, out_channels=transformer_depth, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(transformer_depth),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(output_size=transformer_context_len),
            nn.Dropout1d(dropout_cnn),
        )
        # Output of CNN is (batch, d_model, transformer_context_len)

        # Use sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(transformer_depth)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_depth,
            dim_feedforward=transformer_depth * 4,
            nhead=nhead,
            dropout=dropout_transformer,
            activation='gelu',
            batch_first=True,
            norm_first=True     # LayerNorm first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        # Trainable mask token (to avoid masking with padding value)
        self.mask_token = nn.Parameter(torch.zeros(transformer_depth))

    def forward(self, x, gen_mask=True):
        # CNN specs encoding. Seq len decreasing
        x = self.cnn_encoder(x)  # (batch, cnn_units, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, cnn_units)
        
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
        model_describe += str(self.pos_encoder) + "\n" * 2
        model_describe += "Transformer Encoder: " + str(self.transformer_encoder) + "\n"
        return model_describe


class SpectrogramMaskedDecoder(nn.Module):
    def __init__(self, input_channels, kernel_size, padding, transformer_depth, nhead,
                 num_decoder_layers, dropout_cnn=0.1, dropout_transformer=0.4, device='cuda'):
        super().__init__()
        self.device = device

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_depth,
            nhead=nhead,
            dropout=dropout_transformer,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output of CNN decoder is (batch, IN_CHANNELS, IN_SEQ_LEN)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=transformer_depth,
                               out_channels=128,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout1d(dropout_cnn),

            # nn.ConvTranspose1d(in_channels=256,
            #                    out_channels=128,
            #                    kernel_size=kernel_size,
            #                    stride=kernel_size,
            #                    padding=padding,
            #                    bias=False),
            # nn.BatchNorm1d(128),
            # nn.GELU(),
            # nn.Dropout1d(dropout_cnn),

            nn.ConvTranspose1d(in_channels=128,
                               out_channels=input_channels,
                               kernel_size=kernel_size,
                               stride=kernel_size,
                               padding=padding,
                               bias=False),
            nn.BatchNorm1d(input_channels),
            nn.GELU(),
            nn.Dropout1d(dropout_cnn),
        )

    def forward(self, dec_in, enc_out):
        # X(dec_in) - decoder masked input (batch, seq_len, d_model);
        # Memory(enc_out) - encoder output (batch, vis_len, d_model).
        x = self.transformer_decoder(dec_in, enc_out)  # (batch, seq, d_model)

        # Final decoding. Increasing of the seq len.
        x = x.permute(0, 2, 1)  # (batch, cnn_units, seq)
        x = self.cnn_decoder(x)
        
        return x

    def __str__(self):
        model_describe = ""
        model_describe += "Transformer Decoder: " + str(self.transformer_decoder) + "\n"
        model_describe += "CNN Decoder: " + str(self.cnn_decoder) + "\n" * 2
        return model_describe


class SpectrogramPreTrainedTransformer(nn.Module):
    def __init__(self, encoder: SpectrogramMaskedEncoder, output_dim: int, dropout=0.2, device='cuda'):
        """
        Pre-trained transformer model: encoder -> classifier.
        :param encoder: part of the trained autoencoder.
        :param output_dim: classes to classify number
        """
        super().__init__()
        self.device = device

        self.encoder = encoder
        depth = encoder.transformer_depth
        nhead = encoder.nhead

        self.attention_pool = MultiHeadPool(depth, nhead, dropout=dropout)
        self.output_proj = nn.Linear(depth, output_dim)

    def forward(self, x):
        x, _, _ = self.encoder(x, gen_mask=False)

        # Attention pooling
        x = self.attention_pool(x)  # (batch, depth)
        
        # Final projection (classification)
        logits = self.output_proj(x)
        return logits

    def __str__(self):
        model_describe = ""
        model_describe += "Encoder: " + str(self.encoder) + "\n"
        model_describe += "Attention pooling: " + str(self.attention_pool) + "\n" * 2
        model_describe += "Output projection: " + str(self.output_proj) + "\n" * 2
        return model_describe
