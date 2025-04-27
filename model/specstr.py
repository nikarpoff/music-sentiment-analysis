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
from random import randint
from torch.nn.functional import pad as torch_pad


def truncate_spec(spec, seq_len):
    """
    Truncates spectrogramms to seq_len length. Length is second dimension.
    """
    spec_len = spec.size(1)
    
    if spec_len >= seq_len:
        start = randint(0, spec_len - seq_len)
        spec = spec[:, start:start + seq_len]
    else:
        pad = seq_len - spec_len
        spec = torch_pad(spec, (0, pad), mode='constant', value=0.)

    return spec


class SpectrogramPureTransformer(nn.Module):
    def __init__(self, d_model: int, output_dim: int, nhead: int, num_layers: int, seq_len: int, output_activation: str = "sigmoid", device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.seq_len = seq_len

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
        x = truncate_spec(x, self.seq_len)  # shape (batch, mel_features, seq_len)
        x = x.permute(0, 2, 1)                 # shape (batch, seq_len, mel_features)

        x = self.transformer_encoder(x)
        x = self.linear(x[:, -1, :])
        y = self.output_activation(x)

        return y


class SpectrogramTransformer(nn.Module):
    def __init__(self, d_model: int, output_dim: int, nhead: int, num_layers: int, seq_len: int, output_activation: str = "sigmoid", device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.device = device

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        print("Building model...")

        # Mel-spectrograms have size 96x(sequence_size). So in_channels=96
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=192, kernel_size=5, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=192, out_channels=d_model, kernel_size=5, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Output of CNN is (batch, d_model, new_sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear = nn.Sequential(
            nn.Linear(self.d_model, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

        print(self.cnn, "\n")
        print(self.transformer_encoder, "\n")
        print(self.linear, "\n")
        print("Model built.\n")

    def forward(self, x):
        x = self.cnn(x)  # cnn features - (batch, d_model, new_sequence_length)
        x = truncate_spec(x, self.seq_len)  # truncate to feed transformer
        x = x.permute(0, 2, 1)              # (batch, seq_len, d_model)

        x = self.transformer_encoder(x)     # get transformer features
        x = self.linear(x[:, -1, :])        # get linear projection of last transformer layer
        y = self.output_activation(x)       # get result

        return y
