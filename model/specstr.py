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


class SpectrogramTransformer(nn.Module):
    def __init__(self, d_model: int, output_dim: int, nhead: int, num_layers: int, output_activation: str = "sigmoid", device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        print("Building model...")

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.linear = nn.Linear(self.d_model, self.output_dim)

        print(self.transformer_encoder, "\n")
        print(self.linear, "\n")
        print("Model built.\n")

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x[:, -1, :])
        y = self.output_activation(x)

        return y
