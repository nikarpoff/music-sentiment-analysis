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


import torch
import numpy as np
import pandas as pd
from collections.abc import Iterator

from random import randint
from ast import literal_eval
from os import path as os_path

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def one_hot_encode_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the labels.

    This encoder used for 2/4/8-moods dataset.
    """
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(labels)
    return one_hot_labels

def multi_label_binarize(labels: pd.Series) -> pd.Series:
    """
    Multi Label Binarization encode the labels.

    This encoder used for all-moods dataset.
    """
    encoder = MultiLabelBinarizer()
    binarized_labels = encoder.fit_transform(labels)
    return binarized_labels

def melspecs_collate_fn(batch):
    """
    Custom collaction function with padding to maximum sequence length inside the batch
    """
    xs, ys = zip(*batch)
    xs = [x.permute(1, 0) for x in xs]  

    # Padding by maximum seq_len.
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.)

    # Return initial dimensions order (batch, channel, time)
    xs_padded = xs_padded.permute(0, 2, 1)
    
    ys = torch.stack(ys)

    return xs_padded, ys


class KFoldSpecsDataLoader(Iterator):
    """
    Load the dataset and create DataLoader objects for train/val folds and testing supsets. Iterable by k-fold loaders.

    Splits train/test as 1 - test_size / test_size (0.8 / 0.2 by default).

    Applyes transformation transform_specs for spectrogramms.
    Truncates/paddes spectrogramms if required (if min_seq_len or max_seq_len provided).
    """
    def __init__(self, dataset_path: str, dataset_name: str, splits: int, target_mode: str, min_seq_len: None, max_seq_len: None,
                pad_value=-90., test_size=0.2, batch_size=32, transform_specs: TransformerMixin = None,
                num_workers=8, random_state=None):
        """
        :param dataset_path: Path to the dataset directory.
        :param dataset_name: Name of the dataset file (depends on moods mode).
        :param splits: Number of folds.
        :param target_mode: Target mode for the labels ('multilabel' or 'onehot'). Multi-label is used for all-moods dataset.
        :param max_seq_len: Constant max sequence length. Spectrograms with another length will be truncated to this length. Optional
        :param min_seq_len: Constant min sequence length. Spectrograms with another length will be padded to this length. Optional
        :param pad_value: Pad value for cases where padding of spectrogram required.
        :param transform_specs: sklearn transformer for preprocessing spectrograms. Optional.
        :param test_size: Proportion of the dataset to include in the test split.
        :param batch_size: Batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param random_state: Random seed for reproducibility.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.full_dataset_path = os_path.join(dataset_path, dataset_name)

        self.transform_specs = transform_specs
        self.random_state = random_state
        self.target_mode = target_mode
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.test_size = test_size
        self.splits = splits

        if num_workers == 0:
            self.prefetch_factor = None
            self.persistent_workers = False
        else:
            self.prefetch_factor = 2
            self.persistent_workers = True

        self.x_train, self.x_test, self.y_train, self.y_test = self._load_train_test()
        kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)

        self.kf_iter = kf.split(self.x_train, self.y_train)
        self.current_fold = 0
        self.start_fold = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_fold < self.start_fold:
            next(self.kf_iter)  # skip
            self.current_fold += 1

        self.current_fold += 1
        # Get next fold.
        train_indices, val_indices = next(self.kf_iter)

        # Get slices of X, y.
        x_train = self.x_train.iloc[train_indices]
        y_train = self.y_train.iloc[train_indices]

        x_val = self.x_train.iloc[val_indices]
        y_val = self.y_train.iloc[val_indices]

        # Build loaders and return them.
        train_loader = self._get_specs_loader(x_train, y_train)
        val_loader = self._get_specs_loader(x_val, y_val)

        return train_loader, val_loader

    def get_train_len(self):
        return len(self.x_train)
    
    def set_start(self, start):
        """
        Required for loading from checkpoints.
        """
        self.start_fold = start

    def get_test_loader(self):
        return self._get_specs_loader(self.x_test, self.y_test)

    def _load_train_test(self) -> tuple:
        """
        Service function for loading and spliting .tsv dataset
        """
        df = pd.read_csv(self.full_dataset_path, sep='\t')

        # Select the target transformation based on the target mode.
        if self.target_mode == "multilabel":
            y = pd.DataFrame(multi_label_binarize(df['tags'].apply(literal_eval)))  # process the labels, apply literal_eval to convert strings to lists
        elif self.target_mode == "onehot":
            y = pd.DataFrame(one_hot_encode_labels(df['tags'].to_frame()))  # process the labels to one-hot vectors
        else:
            raise ValueError("Invalid target mode. Choose 'multilabel' or 'onehot'.")

        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=self.test_size, random_state=self.random_state)

        return x_train, x_test, y_train, y_test

    def _get_specs_loader(self, x, y):
        dataset = MelspecsDataset(x, y, dataset_path=self.dataset_path, transform_specs=self.transform_specs,
                                  max_seq_len=self.max_seq_len, min_seq_len=self.min_seq_len, pad_value=self.pad_value)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=melspecs_collate_fn,
        )
    
    def __len__(self):
        return self.splits


class MelspecsDataset(Dataset):
    """
    Loader of mel-spectrograms dataset.
    """
    def __init__(self, x, y, dataset_path, min_seq_len, max_seq_len, pad_value, transform_specs: TransformerMixin = None):
        self.x = x
        self.y = y
        self.dataset_path = dataset_path
        self.transform_specs = transform_specs
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        row = self.x.iloc[idx]

        spec = np.load(os_path.join(self.dataset_path, row['melspecs_path']))

        if self.min_seq_len is not None and spec.shape[1] < self.min_seq_len:
            spec = self.pad_spec(spec)  # pad spec if required
        elif self.max_seq_len is not None and spec.shape[1] > self.max_seq_len:
            spec = self.truncate_spec(spec)  # truncate spec if required

        # Apply transformation to the mel spectrogram if specified
        if self.transform_specs:
            spec = self.transform_specs.transform(spec)

        spec = torch.from_numpy(spec).float()  # size (num_mels, num_frames)
        label = torch.tensor(self.y.iloc[idx], dtype=torch.float)  # size (num_classes,)
        return spec, label

    def pad_spec(self, spec: np.ndarray) -> np.ndarray:
        """
        Pads spectrogramms to min_seq_len length. Length is second dimension.
        """
        pad_width = ((0, 0), (0, self.min_seq_len - spec.shape[1]))
        return np.pad(spec, pad_width=pad_width, mode='constant', constant_values=self.pad_value)

    def truncate_spec(self, spec: np.ndarray) -> np.ndarray:
        """
        Truncates spectrogramms to max_seq_len length. Length is second dimension.

        Makes random correct choise of start point.
        """
        start = randint(0, spec.shape[1] - self.max_seq_len)
        return spec[:, start:start + self.max_seq_len]
