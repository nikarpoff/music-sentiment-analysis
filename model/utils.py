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
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
    MultilabelPrecision, MultilabelRecall, MultilabelF1Score
)


def train_one_epoch(model, scaler, loss_function, metric_objs, is_multi_label, loader, total_batches, epoch, tb_writer, optimizer):
    running_loss = 0.
    avg_loss = 0.
    report_interval = max(1, total_batches // 10)
    
    # Load metrics
    precision_metric, recall_metric, f1_metric = metric_objs

    start_time = time()

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        
        # Scaled Backward Pass and gradient Clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if is_multi_label:
            labels_true = labels.int()
            labels_pred = (outputs > 0.5).int()
        else:
            labels_true = torch.argmax(labels, dim=1)
            labels_pred = torch.nn.Softmax(dim=1)(outputs)
            labels_pred = torch.argmax(outputs, dim=1)

        precision_metric.update(labels_pred, labels_true)
        recall_metric.update(labels_pred, labels_true)
        f1_metric.update(labels_pred, labels_true)

        # Report 10 times per epoch
        if i % report_interval == report_interval - 1:
            time_per_batch = (time() - start_time) / report_interval
            avg_loss = running_loss / report_interval

            print(f'\t batch [{i + 1}/{total_batches}] - loss: {avg_loss:.5f}\t time per batch: {time_per_batch:.2f}')
            tb_x = epoch * len(loader) + i + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
            running_loss = 0.
            start_time = time()

    return avg_loss, (precision_metric, recall_metric, f1_metric)

def validate_one_epoch(model, loss_function, metric_objs, is_multi_label, loader):
    model.eval()  # Set the model to evaluation mode
    val_len = len(loader)
    precision_metric, recall_metric, f1_metric = metric_objs

    running_loss = 0.

    with torch.no_grad():
        for i, vdata in enumerate(loader):
            inputs, labels = vdata
            inputs = inputs.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            running_loss += loss

            if is_multi_label:
                labels_true = labels.int()
                labels_pred = (outputs > 0.5).int()
            else:
                labels_true = torch.argmax(labels, dim=1)
                labels_pred = torch.nn.Softmax(dim=1)(outputs)
                labels_pred = torch.argmax(outputs, dim=1)
            
            precision_metric.update(labels_pred, labels_true)
            recall_metric.update(labels_pred, labels_true)
            f1_metric.update(labels_pred, labels_true)
    
    val_avg_loss = running_loss / val_len
    return val_avg_loss, (precision_metric, recall_metric, f1_metric)

def train_model(model, model_name, num_classes, save_path, target_mode, train_loader, val_loader, lr, epochs, l2_reg):
    """
    Train model. Save checkpoints and trained model to save_path.
    Correctly works with two losses and tasks types: multilabel classification and single-label classification.

    :param model: Model to train.
    :param model_name: Name of the model (specstr, pure_specstr and e.t.c.).
    :param num_classes: Number of moods to classify.
    :param save_path: Path to save checkpoints and trained model.
    :param task_type: Type of the task: multilabel classification or single-label classification.
    :param train_loader: Loader of train data.
    :param val_loader: Loader of validation data.
    :param lr: learning rate.
    :param epochs: Number of train epochs.
    :param l2_reg: Regularization for optimizer.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    scaler = torch.cuda.amp.GradScaler()

    if target_mode == "onehot":
        precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)
        
        loss_function = torch.nn.CrossEntropyLoss()
        is_multilabel = False
    elif target_mode == "multilabel":
        precision_metric = MultilabelPrecision(num_labels=num_classes, average='micro').to(model.device)
        recall_metric = MultilabelRecall(num_labels=num_classes, average='micro').to(model.device)
        f1_metric = MultilabelF1Score(num_labels=num_classes, average='micro').to(model.device)
        
        loss_function = torch.nn.BCEWithLogitsLoss()
        is_multilabel = True
    else:
        raise ValueError(f"Unknown target mode provided: {target_mode}")

    # Collect metrics.
    metrics = (precision_metric, recall_metric, f1_metric)

    # Initialize writers, vloss, timestamps.
    start_timestamp = datetime.now()
    timestamp = start_timestamp.strftime('%d%m%y_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

    best_vloss = float('inf')
    total_batches = len(train_loader)

    # Train cycle.
    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")

        # Train the model for one epoch.
        model.train(True)
        start_time = time()
        train_avg_loss, metrics = train_one_epoch(model, scaler, loss_function, metrics, is_multilabel, train_loader, total_batches, epoch, writer, optimizer)
        train_time = time() - start_time

        # Compute and write remembered train metrics.
        precision_metric, recall_metric, f1_metric = metrics
        train_precision = precision_metric.compute().item()
        train_recall = recall_metric.compute().item()
        train_f1 = f1_metric.compute().item()

        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Validate model.
        start_time = time()
        val_avg_loss, metrics = validate_one_epoch(model, loss_function, metrics, is_multilabel, val_loader)
        val_time = time() - start_time

        # Compute and write remembered validation metrics.
        precision_metric, recall_metric, f1_metric = metrics
        val_precision = precision_metric.compute().item()
        val_recall = recall_metric.compute().item()
        val_f1 = f1_metric.compute().item()

        writer.add_scalar('Loss/validation', val_avg_loss, epoch + 1)
        writer.add_scalar('Precision/validation', val_precision, epoch)
        writer.add_scalar('Recall/validation', val_recall, epoch)
        writer.add_scalar('F1/validation', val_f1, epoch)

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Remember best validation loss.
        if val_avg_loss < best_vloss:
            best_vloss = val_avg_loss
        
        # Save checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_path, f"{model_name}_checkpoint_{timestamp}_epoch_{epoch + 1}.pth"))

        # Log loss and metrics.
        print(f"\n Epoch {epoch + 1}/{epochs} - Training loss: {train_avg_loss:.3f}; Validation loss: {val_avg_loss:.3f}")
        print(f"\t Training: precision: {train_precision:.3f}\t recall: {train_recall:.3f}\t F1: {train_f1:.3f}")
        print(f"\t Validation: precision: {val_precision:.3f}\t recall: {val_recall:.3f}\t F1: {val_f1:.3f}")
        print(f"Train time: {train_time:.3f}; validation time: {val_time:.3f}, total epoch time: {(train_time + val_time):.3f}\n")
        
        torch.cuda.empty_cache()

    # Get total train time and formate it.
    end_timestamp = datetime.now()
    total_learning_time = (end_timestamp - start_timestamp)
    days = total_learning_time.days
    hours, remainder = divmod(total_learning_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formated_learning_time = f"{days:02d} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

    # Close writer and save trained model. Saved model naming is model_name + moods number + timestamp. Save only weigths.
    writer.close()
    model_save_path = os.path.join(save_path, f"{model_name}_{num_classes}_{end_timestamp.strftime('%d_%m_%Y')}.pth")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}\n\t best validation loss: {best_vloss:.3f}; total learning time: {formated_learning_time}")

def evaluate_model(model, num_classes, target_mode, test_loader):
    if target_mode == "onehot":
        precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)

        loss_function = torch.nn.CrossEntropyLoss()
        is_multilabel = False
    elif target_mode == "multilabel":
        precision_metric = MultilabelPrecision(num_labels=num_classes, average='micro').to(model.device)
        recall_metric = MultilabelRecall(num_labels=num_classes, average='micro').to(model.device)
        f1_metric = MultilabelF1Score(num_labels=num_classes, average='micro').to(model.device)

        loss_function = torch.nn.BCEWithLogitsLoss()
        is_multilabel = True
    else:
        raise ValueError(f"Unknown target mode provided: {target_mode}")
    
    # Collect metrics.
    metrics = (precision_metric, recall_metric, f1_metric)

    print("Evaluating model...")
    model.eval()  # Set the model to evaluation mode

    start_time = time()
    test_avg_loss, metrics = validate_one_epoch(model, loss_function, metrics, is_multilabel, test_loader)
    test_time = time() - start_time

    # Compute and write remembered validation metrics.
    precision_metric, recall_metric, f1_metric = metrics
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    
    # Log loss and metrics.
    print(f"Test loss {test_avg_loss:.3f}\t precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}")
    print(f"Test time: {test_time:.3f}")
