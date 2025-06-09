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
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score
)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config import *


def train_text_sentiment_model(
    model: torch.nn.Module,
    model_name: str,
    save_path: str,
    num_classes: int,
    train_loader,
    val_loader,
    lr: float,
    epochs: int,
    l2_reg: float,
):
    """
    Train text sentiment analysis transformer model.
    """

    start_timestamp = datetime.now()
    timestamp = start_timestamp.strftime(TIMESTAMP_FORMAT)
    writer = SummaryWriter(f'runs/train_{model_name}_{timestamp}')
    report_interval = 5
    
    total_batches = len(train_loader)
    
    # AdamW optimizer. Use weigth decay and adaptive learning rate.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

    precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)

    loss_function = torch.nn.CrossEntropyLoss()
    cuda_scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        train_total_loss = 0.
        running_loss = 0.0
        start_time = time()
        model.train()
        
        print(f"\n Epoch {epoch + 1}")

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
            labels = batch["labels"].to(model.device, non_blocking=True).long()

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)

            # Scaled Backward Pass and gradient Clipping
            cuda_scaler.scale(loss).backward()
            cuda_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            cuda_scaler.step(optimizer)
            cuda_scaler.update()

            running_loss += loss.item()
            
            labels_pred = torch.argmax(outputs, dim=1)

            precision_metric.update(labels_pred, labels)
            recall_metric.update(labels_pred, labels)
            f1_metric.update(labels_pred, labels)

            # Report 20 times per epoch
            if i % report_interval == report_interval - 1:
                time_per_batch = (time() - start_time) / report_interval
                avg_loss = running_loss / report_interval
                train_total_loss += avg_loss

                print(f'\t batch [{i + 1}/{total_batches}] - loss: {avg_loss:.5f}\t time per batch: {time_per_batch:.2f}')
                current_step = epoch * total_batches + i
                writer.add_scalar('Loss/train', avg_loss, current_step)
                running_loss = 0.
                start_time = time()

        # Log metrics.
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        writer.add_scalar('Precision/train', precision, epoch)
        writer.add_scalar('Recall/train', recall, epoch)
        writer.add_scalar('F1/train', f1, epoch)

        print(f"\t Training: precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}\n")
        
        # Validation
        model.eval()
        val_batches = len(val_loader)
        running_loss = 0.

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(model.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
                labels = batch["labels"].to(model.device, non_blocking=True).long()

                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)

                running_loss += loss

                labels_pred = torch.argmax(outputs, dim=1)

                precision_metric.update(labels_pred, labels)
                recall_metric.update(labels_pred, labels)
                f1_metric.update(labels_pred, labels)

        val_avg_loss = running_loss / val_batches

        writer.add_scalar('Loss/validation', val_avg_loss, epoch)

        # Log metrics.
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        writer.add_scalar('Precision/validation', precision, epoch)
        writer.add_scalar('Recall/validation', recall, epoch)
        writer.add_scalar('F1/validation', f1, epoch)

        print(f"\t Epoch: {epoch + 1}\t Train average loss: {train_total_loss / total_batches:.3f}\t Validation average loss: {val_avg_loss:.3f}\n")
        print(f"\t Validation: precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_path, f"{model_name}_checkpoint_{timestamp}_epoch_{epoch + 1}.pth"))

    # Train end!
    writer.close()

    # Get total train time and formate it.
    end_timestamp = datetime.now()
    total_learning_time = (end_timestamp - start_timestamp)
    days = total_learning_time.days
    hours, remainder = divmod(total_learning_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formated_learning_time = f"{days:02d} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

    # Close writer and save trained model. Saved model naming is model_name + moods number + timestamp. Save only weigths.
    model_save_path = os.path.join(save_path, f"{model_name}_{end_timestamp.strftime(TIMESTAMP_FORMAT)}.pth")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}\n\t total learning time: {formated_learning_time}")
    