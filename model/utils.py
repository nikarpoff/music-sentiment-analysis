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
from time import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter


def train_one_epoch(model, loss_function, loader, total_batches, epoch, tb_writer, optimizer):
    running_loss = 0.
    last_loss = 0.
    report_interval = total_batches // 10
    start_time = time()

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
            time_per_batch = (time() - start_time) / report_interval
            last_loss = running_loss / report_interval
            print(f'\t batch [{i + 1}/{total_batches}] - train loss: {last_loss:.5f}; time per batch: {time_per_batch:.3f}')
            tb_x = epoch * len(loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            start_time = time()

    return last_loss

def train_model(model, model_name, save_path, loss_name, train_loader, val_loader, lr, epochs, l2_reg):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    if loss_name == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss_name == "binary_cross_entropy_with_logits":
        loss_function = torch.nn.BCEWithLogitsLoss()

    start_timestamp = datetime.now()
    timestamp = start_timestamp.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

    best_vloss = float('inf')
    total_batches = len(train_loader)
    val_len = len(val_loader)

    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")

        # Train the model for one epoch.
        model.train(True)
        start_time = time()
        train_avg_loss = train_one_epoch(model, loss_function, train_loader, total_batches, epoch, writer, optimizer)
        train_time = time() - start_time

        running_vloss = 0.

        # Validation.
        model.eval()  # Set the model to evaluation mode

        start_time = time()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(model.device, non_blocking=True)
                vlabels = vlabels.to(model.device, non_blocking=True)

                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss
        val_time = time() - start_time
        
        val_avg_loss = running_vloss / val_len
        writer.add_scalar('Loss/validation', val_avg_loss, epoch + 1)

        if val_avg_loss < best_vloss:
            best_vloss = val_avg_loss
        
        # Save checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_path, f"{model_name}_checkpoint_{timestamp}_epoch_{epoch + 1}.pth"))

        # Log loss.
        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {train_avg_loss:.3f}; Validation loss: {val_avg_loss:.3f}; Best validation loss: {best_vloss:.3f}")
        print(f"\t train time: {train_time:.3f}; validation time: {val_time:.3f}, total epoch time: {(train_time + val_time):.3f}")
        torch.cuda.empty_cache()

    writer.close()
    model_save_path = os.path.join(save_path, f"{model_name}_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)

    total_learning_time = (datetime.now() - start_timestamp)
    days = total_learning_time.days
    hours, remainder = divmod(total_learning_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formated_learning_time = f"{days:02d} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"Model saved to {model_save_path}\n\t best validation loss: {best_vloss:.3f}; total learning time: {formated_learning_time}")

def evaluate_model(model, loss_name, test_loader):
    if loss_name == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif loss_name == "binary_cross_entropy_with_logits":
        loss_function = torch.nn.BCEWithLogitsLoss()

    print("Evaluating model...")
    model.eval()  # Set the model to evaluation mode

    running_loss = 0.
    test_len = len(test_loader)
    time_start = time()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            running_loss += loss
    
    total_time = time() - time_start
    test_avg_loss = running_loss / test_len

    # Log loss.
    print(f"Test loss: {test_avg_loss:.3f}; Required time: {total_time:.3f}s")
