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

from random import randint
from ast import literal_eval
from os import path as os_path

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
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

def load_specs_dataset(dataset_path: str, dataset_name: str, device, target_mode: str, min_seq_len: None, max_seq_len: None,
                       pad_value=-90., val_size=0.2, test_size=0.2, batch_size=32, transform_specs: TransformerMixin = None,
                       num_workers=8, random_state=None) -> tuple:
    """
    Load the dataset and create DataLoader objects for training, validation, and testing.
    :param dataset_path: Path to the dataset directory.
    :param dataset_name: Name of the dataset file (depends on moods mode).
    :param device: Device to load the data on ('cuda' or 'cpu').
    :param target_mode: Target mode for the labels ('multilabel' or 'onehot'). Multi-label is used for all-moods dataset.
    :param max_seq_len: Constant max sequence length. Spectrograms with another length will be truncated to this length. Optional
    :param min_seq_len: Constant min sequence length. Spectrograms with another length will be padded to this length. Optional
    :param pad_value: Pad value for cases where padding of spectrogram required.
    :param transform_specs: sklearn transformer for preprocessing spectrograms. Optional.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param test_size: Proportion of the dataset to include in the test split.
    :param batch_size: Batch size for the DataLoader.
    :param num_workers: Number of subprocesses to use for data loading.
    :param random_state: Random seed for reproducibility.

    :return: tuple of DataLoader objects for training, validation, and testing.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = _load_dataset(dataset_path, dataset_name, target_mode=target_mode,
                                                              val_size=val_size, test_size=test_size, random_state=random_state)
    
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False
    else:
        prefetch_factor = 2
        persistent_workers = True

    def _get_specs_loader(x, y, dataset_path):
        dataset = MelspecsDataset(x, y, device=device, dataset_path=dataset_path, transform_specs=transform_specs,
                                  max_seq_len=max_seq_len, min_seq_len=min_seq_len, pad_value=pad_value)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            collate_fn=melspecs_collate_fn,
        )

    train_loader = _get_specs_loader(x_train, y_train, dataset_path)
    val_loader = _get_specs_loader(x_val, y_val, dataset_path)
    test_loader = _get_specs_loader(x_test, y_test, dataset_path)

    return train_loader, val_loader, test_loader

def _load_dataset(dataset_path: str, dataset_name: str, target_mode: str, val_size=0.2, test_size=0.2, random_state=None) -> tuple:
    """
    Service function for loading and spliting .tsv dataset
    """
    df = pd.read_csv(os_path.join(dataset_path, dataset_name), sep='\t')

    # Select the target transformation based on the target mode.
    if target_mode == "multilabel":
        y = pd.DataFrame(multi_label_binarize(df['tags'].apply(literal_eval)))  # process the labels, apply literal_eval to convert strings to lists
    elif target_mode == "onehot":
        y = pd.DataFrame(one_hot_encode_labels(df['tags'].to_frame()))  # process the labels to one-hot vectors
    else:
        raise ValueError("Invalid target mode. Choose 'multilabel' or 'onehot'.")
    
    df.drop(columns=['tags'])  # remove the labels from the features

    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state)

    return x_train, y_train, x_val, y_val, x_test, y_test

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


class MelspecsDataset(Dataset):
    """
    Loader of mel-spectrograms dataset.
    """
    def __init__(self, x, y, device, dataset_path, min_seq_len, max_seq_len, pad_value, transform_specs: TransformerMixin = None):
        self.x = x
        self.y = y
        self.dataset_path = dataset_path
        self.transform_specs = transform_specs
        self.device = device
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