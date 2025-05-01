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
import re
import torch
from time import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
    MultilabelPrecision, MultilabelRecall, MultilabelF1Score
)


TIMESTAMP_FORMAT = "%d%m%y_%H%M%S"
DATE_FORMAT = "%d_%m_%y"


class ClassificationModelTrainer():
    """
    Model trainer. Save checkpoints and trained model to save_path.
    Correctly works with two losses and tasks types: multilabel classification and single-label classification.
    """
    def __init__(self, model, model_name, num_classes, save_path, target_mode, train_loader, val_loader, lr, epochs, l2_reg):
        """
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
        self.model = model
        self.model_name = model_name
        self.num_classes = num_classes
        self.save_path = save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.epoch = 0
        self.best_vloss = float('inf')
        self.train_batches = len(train_loader)
        self.val_batches = len(val_loader)
        self.report_interval = max(1, self.train_batches // 20)
        
        # AdamW optimizer. Use weigth decay and adaptive learning rate.
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)

        # Sheduler 1: OneCycleLR (per batch)
        total_steps = epochs * len(train_loader)
        self.sheduler_one_cycle = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr * 10,
            total_steps=total_steps,
            pct_start=0.3,          # 30% for warm-up
            div_factor=1e3,         # div factor for start lr
            final_div_factor=1e4,   # div factor for final lr
            anneal_strategy='cos'   # cos strategy for decrease lr
        )

        # Scheduler 2: ReduceLROnPlateau (per epoch)
        self.sheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',     # minimize loss
            factor=0.5,     # div factor for lr
            patience=2,     # epochs number without improvement (after decrease lr)
            min_lr=1e-7     # min available lr
        )

        self.cuda_scaler = torch.cuda.amp.GradScaler()

        # Select metrics, loss by type of classification task.
        if target_mode == "onehot":
            self.precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
            self.recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
            self.f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)

            self.loss_function = torch.nn.CrossEntropyLoss()
            self.is_multilabel = False
        elif target_mode == "multilabel":
            self.precision_metric = MultilabelPrecision(num_labels=num_classes, average='micro').to(model.device)
            self.recall_metric = MultilabelRecall(num_labels=num_classes, average='micro').to(model.device)
            self.f1_metric = MultilabelF1Score(num_labels=num_classes, average='micro').to(model.device)

            self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.is_multilabel = True
        else:
            raise ValueError(f"Unknown target mode provided: {target_mode}")
        
        self.start_timestamp = None
        self.timestamp = None
        self.date = None
        self.writer = None

    def init_new_train(self):
        # Initialize writers, timestamps.
        self.start_timestamp = datetime.now()
        self.date = self.start_timestamp.strftime(DATE_FORMAT)
        self.timestamp = self.start_timestamp.strftime(TIMESTAMP_FORMAT)
        self.writer = SummaryWriter(f'runs/train_{self.model_name}_{self.num_classes}_{self.timestamp}')

    def init_continue_train(self, saved_model_name):
        saved_model_path = os.path.join(self.save_path, saved_model_name)

        if not os.path.isfile(saved_model_path):
            raise FileNotFoundError(f"Not found model {saved_model_name} by path {self.save_path}!")
        
        if "checkpoint" in saved_model_name:
            ckpt = torch.load(saved_model_path, map_location=self.model.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.sheduler_one_cycle.load_state_dict(ckpt['one_cycle_state_dict'])
            self.sheduler_plateau.load_state_dict(ckpt['plateau_state_dict'])
            self.epoch = ckpt['epoch']

            match = re.search(r"checkpoint_(\d{6})_(\d{6})_epoch_(\d+)\.", saved_model_name)

            if match:
                date_str, time_str, _ = match.groups()
                self.start_timestamp = datetime.strptime(f"{date_str}_{time_str}", TIMESTAMP_FORMAT)
            else:
                print(f"Invalid checkpoint file name: expected '{self.model_name}_checkpoint_<{DATE_FORMAT}>_epoch_<number>', found: {saved_model_name}")
                self.start_timestamp = datetime.now()
        else:
            self.model.load_state_dict(torch.load(saved_model_path, weights_only=True))
            self.start_timestamp = datetime.now()

        self.date = self.start_timestamp.strftime(DATE_FORMAT)
        self.timestamp = self.start_timestamp.strftime(TIMESTAMP_FORMAT)
        self.writer = SummaryWriter(f'runs/train_{self.model_name}_{self.num_classes}_{self.timestamp}')
        print(f"Loaded model:\n", self.model)

    def train_model(self):
        if self.start_timestamp is None:
            print("First, call init_new_train()/init_continue_train()!")
            return

        while self.epoch < self.epochs:
            print(f"Epoch {self.epoch + 1}/{self.epochs}")

            # Train for one epoch.
            start_time = time()
            train_avg_loss = self._train_one_epoch()
            epoch_train_time = time() - start_time
            
            train_precision, train_recall, train_f1 = self._compute_and_reset_metrics()
            
            self.writer.add_scalar('Precision/train', train_precision, self.epoch + 1)
            self.writer.add_scalar('Recall/train', train_recall, self.epoch + 1)
            self.writer.add_scalar('F1/train', train_f1, self.epoch + 1)

            # Validate model.
            start_time = time()
            val_avg_loss = self._validate_one_epoch()
            epoch_val_time = time() - start_time
            
            val_precision, val_recall, val_f1 = self._compute_and_reset_metrics()

            self.writer.add_scalar('Loss/validation', val_avg_loss, self.epoch + 1)
            self.writer.add_scalar('Precision/validation', val_precision, self.epoch + 1)
            self.writer.add_scalar('Recall/validation', val_recall, self.epoch + 1)
            self.writer.add_scalar('F1/validation', val_f1, self.epoch + 1)

            # Remember best validation loss.
            if val_avg_loss < self.best_vloss:
                self.best_vloss = val_avg_loss

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning rate', current_lr, self.epoch + 1)

            # Save checkpoint.
            torch.save({
                'epoch': self.epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'one_cycle_state_dict': self.sheduler_one_cycle.state_dict(),
                'plateau_state_dict': self.sheduler_plateau.state_dict(),
            }, os.path.join(self.save_path, f"{self.model_name}_checkpoint_{self.timestamp}_epoch_{self.epoch + 1}.pth"))

            # Log loss and metrics.
            print(f"\n Epoch {self.epoch + 1} - Training loss: {train_avg_loss:.3f}; Validation loss: {val_avg_loss:.3f}; lr: {current_lr:.2e}")
            print(f"\t Training: precision: {train_precision:.3f}\t recall: {train_recall:.3f}\t F1: {train_f1:.3f}")
            print(f"\t Validation: precision: {val_precision:.3f}\t recall: {val_recall:.3f}\t F1: {val_f1:.3f}")
            print(f"Train time: {epoch_train_time:.3f}; validation time: {epoch_val_time:.3f}, total epoch time: {(epoch_train_time + epoch_val_time):.3f}\n")

            torch.cuda.empty_cache()
            self.epoch += 1
        
        # Train end!
        self.writer.close()

        # Get total train time and formate it.
        end_timestamp = datetime.now()
        total_learning_time = (end_timestamp - self.start_timestamp)
        days = total_learning_time.days
        hours, remainder = divmod(total_learning_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formated_learning_time = f"{days:02d} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

        # Close writer and save trained model. Saved model naming is model_name + moods number + timestamp. Save only weigths.
        model_save_path = os.path.join(self.save_path, f"{self.model_name}_{self.num_classes}_{end_timestamp.strftime(DATE_FORMAT)}.pth")
        torch.save(self.model.state_dict(), model_save_path)

        print(f"Model saved to {model_save_path}\n\t best validation loss: {self.best_vloss:.3f}; total learning time: {formated_learning_time}")

    def _compute_and_reset_metrics(self):
        precision = self.precision_metric.compute().item()
        recall = self.recall_metric.compute().item()
        f1 = self.f1_metric.compute().item()

        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        return precision, recall, f1

    def _train_one_epoch(self):
        self.model.train(True)
        running_loss = 0.
        avg_loss = 0.

        start_time = time()

        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs = inputs.to(self.model.device, non_blocking=True)
            labels = labels.to(self.model.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)

            # Scaled Backward Pass and gradient Clipping
            self.cuda_scaler.scale(loss).backward()
            self.cuda_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.cuda_scaler.step(self.optimizer)
            self.cuda_scaler.update()

            # Use lr sheduler.
            self.sheduler_one_cycle.step()

            running_loss += loss.item()

            if self.is_multilabel:
                labels_true = labels.int()
                labels_pred = (outputs > 0.5).int()
            else:
                labels_true = torch.argmax(labels, dim=1)
                labels_pred = torch.nn.Softmax(dim=1)(outputs)
                labels_pred = torch.argmax(outputs, dim=1)

            self.precision_metric.update(labels_pred, labels_true)
            self.recall_metric.update(labels_pred, labels_true)
            self.f1_metric.update(labels_pred, labels_true)

            # Report 20 times per epoch
            if i % self.report_interval == self.report_interval - 1:
                time_per_batch = (time() - start_time) / self.report_interval
                avg_loss = running_loss / self.report_interval

                print(f'\t batch [{i + 1}/{self.train_batches}] - loss: {avg_loss:.5f}\t time per batch: {time_per_batch:.2f}')
                tb_x = self.epoch * self.train_batches + i + 1
                self.writer.add_scalar('Loss/train', avg_loss, tb_x)
                running_loss = 0.
                start_time = time()

        return avg_loss

    def _validate_one_epoch(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs = inputs.to(self.model.device, non_blocking=True)
                labels = labels.to(self.model.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                running_loss += loss

                if self.is_multilabel:
                    labels_true = labels.int()
                    labels_pred = (outputs > 0.5).int()
                else:
                    labels_true = torch.argmax(labels, dim=1)
                    labels_pred = torch.nn.Softmax(dim=1)(outputs)
                    labels_pred = torch.argmax(outputs, dim=1)

                self.precision_metric.update(labels_pred, labels_true)
                self.recall_metric.update(labels_pred, labels_true)
                self.f1_metric.update(labels_pred, labels_true)

        val_avg_loss = running_loss / self.val_batches
        return val_avg_loss


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
    
    print("Evaluating model...")
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.
    start_time = time()
    
    # Testing.
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            running_loss += loss

            if is_multilabel:
                labels_true = labels.int()
                labels_pred = (outputs > 0.5).int()
            else:
                labels_true = torch.argmax(labels, dim=1)
                labels_pred = torch.nn.Softmax(dim=1)(outputs)
                labels_pred = torch.argmax(outputs, dim=1)

            precision_metric.update(labels_pred, labels_true)
            recall_metric.update(labels_pred, labels_true)
            f1_metric.update(labels_pred, labels_true)

    test_avg_loss = running_loss / len(test_loader)
    test_time = time() - start_time

    # Compute and write remembered validation metrics.
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    
    # Log loss and metrics.
    print(f"Test loss {test_avg_loss:.3f}\t precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}")
    print(f"Test time: {test_time:.3f}")
