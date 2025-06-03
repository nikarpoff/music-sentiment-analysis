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
from transformers import Wav2Vec2Model


class RawAudioTransformer(nn.Module):
    def __init__(self, output_channels: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        CNN_UNITS = [256, 512, 512, 1024, 1024, 512, 256]
        CNN_KERNELS = [10, 5, 3, 3, 3, 3, 3]
        CNN_STRIDES = [7, 3, 2, 2, 2, 2, 2]
        CNN_RES_CON = [False, False, True, True, True, True, True]
        CNN_PADDINGS = [0] * len(CNN_UNITS)
        RNN_UNITS = 256
        RNN_LAYERS = 2
        TRANSFORMER_DEPTH = 432
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


class PretrainedRawAudioTransformer(nn.Module):
    def __init__(self, output_channels: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
        )

        # Frozen parameters better and faster training
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        hidden_size = self.wav2vec.config.hidden_size

        self.attention_pooling = MultiHeadPool(
            d_model=hidden_size,
            nhead=8,
            dropout=dropout
        )

        self.output_proj = nn.Linear(hidden_size, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.wav2vec(x.squeeze(1)).last_hidden_state
        pooled = self.attention_pooling(features)
        return self.output_proj(pooled)
