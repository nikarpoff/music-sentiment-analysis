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

import os
import torch
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter


def train_one_epoch(model, loss_function, loader, total_batches, epoch, tb_writer, optimizer):
    running_loss = 0.
    last_loss = 0.
    report_interval = total_batches // 10

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        # Report 10 times per epoch
        if i % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            print(f'\t batch [{i + 1}/{total_batches}] - train loss: {last_loss:.5f}')
            tb_x = epoch * len(loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train_model(model, save_path, loss_name, train_loader, val_loader, lr, epochs, l2_reg):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    if loss_name == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss_name == "binary_cross_entropy_with_logits":
        loss_function = torch.nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = float('inf')
    total_batches = len(train_loader)
    val_len = len(val_loader)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train the model for one epoch.
        model.train(True)
        train_avg_loss = train_one_epoch(model, loss_function, train_loader, total_batches, epoch, writer, optimizer)
        running_vloss = 0.

        # Validation.
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(model.device, non_blocking=True)
                vlabels = vlabels.to(model.device, non_blocking=True)

                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss
        
        val_avg_loss = running_vloss / val_len
        writer.add_scalar('Loss/validation', val_avg_loss, epoch + 1)

        if val_avg_loss < best_vloss:
            best_vloss = val_avg_loss
        
        # Save checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_path, f"checkpoint_{timestamp}_epoch_{epoch + 1}.pth"))

        # Log loss.
        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {train_avg_loss:.3f}; Validation loss: {val_avg_loss:.3f}. Best validation loss: {best_vloss:.3f}")
        torch.cuda.empty_cache()

    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path, f"model_{timestamp}.pth"))

def evaluate_model(model, loss_function, test_loader):
    pass