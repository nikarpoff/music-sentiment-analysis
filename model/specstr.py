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
from torch import no_grad
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter


class SpectrogramTransformer(nn.Module):
    def __init__(self, d_model: int, output_dim: int, nhead: int, num_layers: int, seq_len: int, target_mode: str):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.model = None
        
        # Select the target transformation based on the target mode.
        if target_mode == "multi_label":
            self.output_activation = nn.Sigmoid()
            self.loss = F.binary_cross_entropy_with_logits
        elif target_mode == "one_hot":
            self.output_activation = nn.Softmax(dim=1)
            self.loss = F.cross_entropy
        else:
            raise ValueError("Invalid target mode. Choose 'multi_label' or 'one_hot'.")

    def build_model(self):
        print("Building model...")

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        model = nn.Sequential(
            transformer_encoder,
            nn.Linear(self.d_model, self.output_dim),
            self.output_activation
        )

        print(model, "\n")
        print("Model built.\n")

        self.model = model

    def _forward(self, x):
        return self.model(x)

    def _train_one_epoch(self, loader, total_batches, epoch, tb_writer, optimizer):
        running_loss = 0.
        last_loss = 0.
        report_interval = total_batches // 10

        for i, data in enumerate(loader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # Report 10 times per epoch
            if i % report_interval == report_interval - 1:
                last_loss = running_loss / report_interval
                print(f'\t batch [{i + 1}/{total_batches}] - train loss: {last_loss}')
                tb_x = epoch * len(loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self, train_loader, val_loader, lr, epochs, l2_reg):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=l2_reg)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        best_vloss = float('inf')

        total_batches = len(train_loader)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train(True)

            # Train the model for one epoch.
            train_avg_loss = self._train_one_epoch(train_loader, total_batches, epoch, writer, optimizer)

            running_vloss = 0.
            self.model.eval()  # Set the model to evaluation mode

            # Validation.
            with no_grad():
                for _, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss(voutputs, vlabels)
                    running_vloss += vloss

            if running_vloss < best_vloss:
                best_vloss = running_vloss

            # Log loss.
            print(f"Epoch {epoch + 1}/{epochs} - Training loss: {train_avg_loss}; Validation loss: {running_vloss}")

    def evaluate(self, test_loader):
        pass
