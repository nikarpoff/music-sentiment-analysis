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
from model.data import KFoldSpecsDataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
    MultilabelPrecision, MultilabelRecall, MultilabelF1Score
)

from config import *


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        :param patience: epoch count, after which training will be stopped if no improvement
        :param min_delta: minimal loss delta to consider improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss: float):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelTrainer():
    def __init__(self, model, model_name: str, save_path: str, kfold_loader: KFoldSpecsDataLoader, lr: float, epochs: int, l2_reg: float):
        """
        :param model: Model to train.
        :param model_name: Name of the model (specstr, pure_specstr and e.t.c.).
        :param save_path: Path to save checkpoints and trained model.
        :param kfold_loader: Loader of train data. Iterable object with train/val loaders as elements.
        :param lr: learning rate.
        :param epochs: Number of train epochs.
        :param l2_reg: Regularization for optimizer.
        """
        self.model = model
        self.model_name = model_name
        self.save_path = save_path
        self.kfold_loader = kfold_loader
        self.early_stopper = EarlyStopping(patience=5, min_delta=1e-4)
        self.epochs = epochs

        self.fold = 0
        self.epoch = 0
        self.best_vloss = float('inf')
        self.folds = len(kfold_loader)
        self.report_times = 20
        self.l2_reg = l2_reg
        self.lr = lr
    
        self.cuda_scaler = torch.amp.GradScaler("cuda")

        self.start_timestamp = None
        self.timestamp = None
        self.date = None
        self.writer = None

    def init_new_train(self):
        # Initialize writers, timestamps.
        self.start_timestamp = datetime.now()
        self.date = self.start_timestamp.strftime(DATE_FORMAT)
        self.timestamp = self.start_timestamp.strftime(TIMESTAMP_FORMAT)
        self.writer = SummaryWriter(f'runs/train_{self.model_name}_{self.timestamp}')

    def init_continue_train(self, saved_model_name):
        saved_model_path = os.path.join(self.save_path, saved_model_name)

        if not os.path.isfile(saved_model_path):
            raise FileNotFoundError(f"Not found model {saved_model_name} by path {self.save_path}!")
        
        if "checkpoint" in saved_model_name:
            ckpt = torch.load(saved_model_path, map_location=self.model.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self._recreate_optimizer_and_shedulers(1)

            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.sheduler_one_cycle.load_state_dict(ckpt["one_cycle_state_dict"])
            self.epoch = ckpt["epoch"] + 1
            self.fold = ckpt["fold"]

            if self.epoch == self.epochs:
                self.fold += 1
                self.epoch = 0

            self.kfold_loader.set_start(self.fold)  # start loading folds from checkpoint's fold

            match = re.search(r".*?(\d{6})_(\d{6}).*", saved_model_name)

            if match:
                date_str, time_str = match.groups()
                self.start_timestamp = datetime.strptime(f"{date_str}_{time_str}", TIMESTAMP_FORMAT)
            else:
                print(f"Invalid checkpoint file name: expected '{self.model_name}_checkpoint_<{TIMESTAMP_FORMAT}>_fold_<number>_epoch_<number>', found: {saved_model_name}")
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
        
        # For every fold.
        for train_loader, val_loader in self.kfold_loader:
            # If epoch is 0 (we start not from checkpoint) then recreate shedulers and AdamW
            if self.epoch == 0:
                self._recreate_optimizer_and_shedulers(len(train_loader))

            # And for every epoch.
            while self.epoch < self.epochs:
                print(f"Fold {self.fold + 1}/{self.folds}; Epoch {self.epoch + 1}/{self.epochs}")
                current_iteration = (self.fold * self.epochs) + self.epoch + 1

                # Train for one epoch.
                start_time = time()
                train_avg_loss = self._train_one_epoch(train_loader, current_iteration)
                epoch_train_time = time() - start_time

                # Validate model.
                start_time = time()
                val_avg_loss = self._validate_one_epoch(val_loader, current_iteration)
                epoch_val_time = time() - start_time

                self.writer.add_scalar('Loss/validation', val_avg_loss, current_iteration)

                # Remember best validation loss.
                if val_avg_loss < self.best_vloss:
                    self.best_vloss = val_avg_loss

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning rate', current_lr, current_iteration)

                # Save checkpoint.
                torch.save({
                    'fold': self.fold,
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'one_cycle_state_dict': self.sheduler_one_cycle.state_dict(),
                }, os.path.join(self.save_path, f"{self.model_name}_checkpoint_{self.timestamp}_fold_{self.fold + 1}_epoch_{self.epoch + 1}.pth"))

                # Log loss and metrics.
                print(f"\n Fold {self.fold + 1}; Epoch {self.epoch + 1} - Training loss: {train_avg_loss:.3f}; Validation loss: {val_avg_loss:.3f}; lr: {current_lr:.2e}")
                print(f"Train time: {epoch_train_time:.3f}; validation time: {epoch_val_time:.3f}, total epoch time: {(epoch_train_time + epoch_val_time):.3f}\n")

                torch.cuda.empty_cache()

                # Early stopping.
                self.early_stopper.step(val_avg_loss)
                if self.early_stopper.should_stop:
                    print(f"Early stopping: no improvement for {self.early_stopper.patience} epochs.")
                    break
                
                self.epoch += 1

            self.fold += 1
            self.epoch = 0
        
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
        model_save_path = os.path.join(self.save_path, f"{self.model_name}_{end_timestamp.strftime(DATE_FORMAT)}.pth")
        torch.save(self.model.state_dict(), model_save_path)

        print(f"Model saved to {model_save_path}\n\t best validation loss: {self.best_vloss:.3f}; total learning time: {formated_learning_time}")

    def _recreate_optimizer_and_shedulers(self, epoch_steps) -> None:
        # AdamW optimizer. Use weigth decay and adaptive learning rate.
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        # Sheduler: OneCycleLR (per batch)
        fold_steps = self.epochs * epoch_steps
        self.sheduler_one_cycle = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr * 5,
            total_steps=fold_steps,
            pct_start=0.3,          # 30% for warm-up
            div_factor=10,          # div factor for start lr
            final_div_factor=1e4,   # div factor for final lr
            anneal_strategy='cos'   # cos strategy for decrease lr
        )

        # Set early stoping epochs without improving counter to zero.
        self.early_stopper.counter = 0

    def _train_one_epoch(self, loader, current_iteration):
        pass

    def _validate_one_epoch(self, loader, current_iteration):
        pass


class ClassificationModelTrainer(ModelTrainer):
    """
    Model trainer. Save checkpoints and trained model to save_path.
    Correctly works with two losses and tasks types: multilabel classification, single-label classification, autoencoder regression
    """
    def __init__(self, model, model_name: str, save_path: str, target_mode: str, kfold_loader: KFoldSpecsDataLoader, lr: float, epochs: int, l2_reg: float, num_classes: int = None):
        """
        :param num_classes: Number of moods to classify. If None -> target_mode should not be classification.
        :param task_type: Type of the task: multilabel classification, single-label classification or autoencoder regression.
        """
        super().__init__(model, model_name, save_path, kfold_loader, lr, epochs, l2_reg)
        self.target_mode = target_mode

        # Select metrics, loss by type of classification task.
        if target_mode == ONE_HOT_TARGET:
            self.precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
            self.recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
            self.f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)

            self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            self.is_multilabel = False
        elif target_mode == MULTILABEL_TARGET:
            self.precision_metric = MultilabelPrecision(num_labels=num_classes, average='micro').to(model.device)
            self.recall_metric = MultilabelRecall(num_labels=num_classes, average='micro').to(model.device)
            self.f1_metric = MultilabelF1Score(num_labels=num_classes, average='micro').to(model.device)

            self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.is_multilabel = True
        elif target_mode == AUTOENCODER_TARGET:
            self.loss_function = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown target mode provided: {target_mode}")

    def _compute_and_reset_metrics(self):
        precision = self.precision_metric.compute().item()
        recall = self.recall_metric.compute().item()
        f1 = self.f1_metric.compute().item()

        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        return precision, recall, f1

    def _train_one_epoch(self, loader, current_iteration):
        total_batches = len(loader)
        report_interval = max(1, total_batches // self.report_times)
        self.model.train(True)
        running_loss = 0.
        avg_loss = 0.

        start_time = time()

        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(self.model.device, non_blocking=True)
            labels = labels.to(self.model.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
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

            self.update_classification_metrics(outputs, labels)

            # Report 20 times per epoch
            if i % report_interval == report_interval - 1:
                time_per_batch = (time() - start_time) / report_interval
                avg_loss = running_loss / report_interval

                print(f'\t batch [{i + 1}/{total_batches}] - loss: {avg_loss:.5f}\t time per batch: {time_per_batch:.2f}')
                current_step = ((self.fold * self.epochs) + self.epoch) * total_batches + i
                self.writer.add_scalar('Loss/train', avg_loss, current_step)
                running_loss = 0.
                start_time = time()

        # Log metrics.
        precision, recall, f1 = self._compute_and_reset_metrics()

        self.writer.add_scalar('Precision/train', precision, current_iteration)
        self.writer.add_scalar('Recall/train', recall, current_iteration)
        self.writer.add_scalar('F1/train', f1, current_iteration)

        print(f"\t Training: precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}\n")
        return avg_loss

    def _validate_one_epoch(self, loader, current_iteration):
        self.model.eval()  # Set the model to evaluation mode
        val_batches = len(loader)
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.to(self.model.device, non_blocking=True)
                labels = labels.to(self.model.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)

                running_loss += loss
                self.update_classification_metrics(outputs, labels)

        val_avg_loss = running_loss / val_batches

        # Log metrics.
        precision, recall, f1 = self._compute_and_reset_metrics()

        self.writer.add_scalar('Precision/validation', precision, current_iteration)
        self.writer.add_scalar('Recall/validation', recall, current_iteration)
        self.writer.add_scalar('F1/validation', f1, current_iteration)

        print(f"\t Validation: precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}\n")
        return val_avg_loss


    def update_classification_metrics(self, model_output, labels):
        if self.is_multilabel:
            labels_true = labels.int()
            labels_pred = (model_output > 0.5).int()
        else:
            labels_true = torch.argmax(labels, dim=1)
            labels_pred = torch.nn.Softmax(dim=1)(model_output)
            labels_pred = torch.argmax(model_output, dim=1)

        self.precision_metric.update(labels_pred, labels_true)
        self.recall_metric.update(labels_pred, labels_true)
        self.f1_metric.update(labels_pred, labels_true)


class AutoencoderModelTrainer(ModelTrainer):
    """
    Model trainer. Save checkpoints and trained model to save_path.
    Correctly works with autoencoder regression
    """
    def __init__(self, model, model_name: str, save_path: str, kfold_loader: KFoldSpecsDataLoader, lr: float, epochs: int, l2_reg: float):
        super().__init__(model, model_name, save_path, kfold_loader, lr, epochs, l2_reg)
        self.loss_function = torch.nn.MSELoss()

    def _train_one_epoch(self, loader, current_iteration):
        total_batches = len(loader)
        report_interval = max(1, total_batches // self.report_times)
        self.model.train(True)
        running_loss = 0.
        avg_loss = 0.

        start_time = time()

        for i, data in enumerate(loader):
            inputs, _ = data
            inputs = inputs.to(self.model.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.model(inputs)
                loss = self.loss_function(inputs, outputs)

            # Scaled Backward Pass and gradient Clipping
            self.cuda_scaler.scale(loss).backward()
            self.cuda_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.cuda_scaler.step(self.optimizer)
            self.cuda_scaler.update()

            # Use lr sheduler.
            self.sheduler_one_cycle.step()

            running_loss += loss.item()

            # Report 20 times per epoch
            if i % report_interval == report_interval - 1:
                time_per_batch = (time() - start_time) / report_interval
                avg_loss = running_loss / report_interval

                print(f'\t batch [{i + 1}/{total_batches}] - loss: {avg_loss:.5f}\t time per batch: {time_per_batch:.2f}')
                current_step = ((self.fold * self.epochs) + self.epoch) * total_batches + i
                self.writer.add_scalar('Loss/train', avg_loss, current_step)
                running_loss = 0.
                start_time = time()

        return avg_loss

    def _validate_one_epoch(self, loader, current_iteration):
        self.model.eval()  # Set the model to evaluation mode
        val_batches = len(loader)
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, _ = data
                inputs = inputs.to(self.model.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.loss_function(inputs, outputs)

                running_loss += loss

        val_avg_loss = running_loss / val_batches
        return val_avg_loss


def evaluate_classification_model(model, num_classes, target_mode, test_loader):
    if target_mode == ONE_HOT_TARGET:
        precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(model.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(model.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(model.device)

        loss_function = torch.nn.CrossEntropyLoss()
        is_multilabel = False
    elif target_mode == MULTILABEL_TARGET:
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

    # Log loss and time.
    print(f"Test time: {test_time:.3f}\t loss: {test_avg_loss:.3f}")

    # Compute and write remembered test metrics.
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    
    # Log metrics.
    print(f"Test precision: {precision:.3f}\t recall: {recall:.3f}\t F1: {f1:.3f}")



def evaluate_autoencoder(model, test_loader):    
    print("Evaluating model...")
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.
    start_time = time()
    loss_function = torch.nn.MSELoss()
    
    # Testing.
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, _ = data
            inputs = inputs.to(model.device, non_blocking=True)

            outputs = model(inputs)
            loss = loss_function(inputs, outputs)

            running_loss += loss

    test_avg_loss = running_loss / len(test_loader)
    test_time = time() - start_time

    # Log loss and time.
    print(f"Test time: {test_time:.3f}\t loss: {test_avg_loss:.3f}")


def get_params_count(model: torch.nn.Module) -> tuple:
    """
    Get number of parameters in the model.
    :return: tuple (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
