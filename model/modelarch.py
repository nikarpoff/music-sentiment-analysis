import torch
import torch.nn as nn
from training.model.layer import ConGRUFormer


class TinyRawAudioTransformer(nn.Module):
    def __init__(self, output_channels: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_UNITS = [128, 256, 512, 1024, 512, 256]
        CNN_KERNELS = [10, 5, 3, 3, 3, 3]
        CNN_STRIDES = [7, 3, 2, 2, 2, 2]
        CNN_RES_CON = [False, False, True, True, True, True]
        CNN_PADDINGS = [0] * len(CNN_UNITS)
        RNN_UNITS = 256
        RNN_LAYERS = 2
        TRANSFORMER_DEPTH = 312
        NHEAD = 6
        NUM_ENCODERS = 4

        TRANSFORMER_DROPOUT = 0.3

        self.congruformer = ConGRUFormer(
            in_channels=1,
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

        self.output_proj = nn.Linear(TRANSFORMER_DEPTH, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.congruformer(x)
        return self.output_proj(x)

class SpectrogramTransformer(nn.Module):
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
    