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

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.linear = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x):
        # x = truncate_spec(x, self.seq_len)  # shape (batch, mel_features, seq_len)
        x = x.permute(0, 2, 1)                 # shape (batch, seq_len, mel_features)

        x = self.transformer_encoder(x)
        x = self.linear(x[:, -1, :])
        y = self.output_activation(x)

        return y

    def __str__(self):
        model_describe = ""
        model_describe += str(self.transformer_encoder) + "\n"
        model_describe += str(self.linear) + "\n"
        model_describe += str(self.output_activation) + "\n"
        return model_describe

class SpectrogramTransformer(nn.Module):
    def __init__(self, cnn_units: int, rnn_units: int, d_model: int, output_dim: int, nhead: int, num_layers: int,
                 seq_len: int, output_activation: str = "sigmoid", dropout=0.2, device='cuda'):
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

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        # Mel-spectrograms have size 96x(sequence_size). So in_channels=96
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=self.cnn_units, kernel_size=5, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Output of CNN is (batch, d_model, new_sequence_length)

        self.rnn = nn.GRU(self.cnn_units, rnn_units, num_layers=2, batch_first=True, dropout=self.dropout)

        self.d_model_proj = nn.Sequential(
            nn.Linear(rnn_units, d_model),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.output_dim)
        )

    def forward(self, x):
        x = self.cnn(x)  # cnn features - (batch, cnn_units, cnn_sequence_length)

        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)  # (batch, rnn_sequence_length, gru_units)

        x = self.d_model_proj(x)  # (batch, rnn_sequence_length, d_model)

        x = self.transformer_encoder(x)     # get transformer features
        x = self.output_proj(x[:, -1, :])   # get linear projection of last transformer layer
        y = self.output_activation(x)       # get result

        return y

    def __str__(self):
        model_describe = ""
        model_describe += str(self.cnn) + "\n" * 2
        model_describe += str(self.rnn) + "\n" * 2
        model_describe += str(self.d_model_proj) + "\n" * 2
        model_describe += str(self.transformer_encoder) + "\n"
        model_describe += str(self.output_proj) + "\n" * 2
        model_describe += str(self.output_activation) + "\n"
        return model_describe
