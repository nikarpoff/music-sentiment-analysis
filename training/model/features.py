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


class FeaturesDense(nn.Module):
    def __init__(self, input_channels: 64, output_channels: int, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Linear(64, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.output_proj(x)
    