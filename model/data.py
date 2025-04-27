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

import numpy as np
import pandas as pd

from ast import literal_eval
from os import path as os_path

from torch import stack
from torch import from_numpy
from torch import float as torch_float
from torch import tensor as torch_tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def one_hot_encode_labels(labels: pd.Series) -> pd.Series:
    """
    One-hot encode the labels.

    This encoder used for 2/4/8-moods dataset.
    """
    encoder = OneHotEncoder()
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

def load_specs_dataset(dataset_path: str, dataset_name: str, device, target_mode: str, val_size=0.2,
                       test_size=0.2, batch_size=32, num_workers=8, random_state=None) -> tuple:
    """
    Load the dataset and create DataLoader objects for training, validation, and testing.
    :param dataset_path: Path to the dataset directory.
    :param dataset_name: Name of the dataset file (depends on moods mode).
    :param device: Device to load the data on ('cuda' or 'cpu').
    :param target_mode: Target mode for the labels ('multi_label' or 'one_hot'). Multi-label is used for all-moods dataset.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param test_size: Proportion of the dataset to include in the test split.
    :param batch_size: Batch size for the DataLoader.
    :param num_workers: Number of subprocesses to use for data loading.
    :param random_state: Random seed for reproducibility.

    :return: tuple of DataLoader objects for training, validation, and testing.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = _load_dataset(dataset_path, dataset_name, target_mode=target_mode,
                                                              val_size=val_size, test_size=test_size, random_state=random_state)
    
    def _get_specs_loader(x, y, dataset_path):
        dataset = MelspecsDataset(x, y, device=device, dataset_path=dataset_path)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # number of subprocesses to use for data loading
            prefetch_factor=2,        # how many batches to prefetch
            persistent_workers=True,  # continue to use the same workers
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
    # Select the target transformation based on the target mode.
    if target_mode == "multi_label":
        transform_target = multi_label_binarize
    elif target_mode == "one_hot":
        transform_target = one_hot_encode_labels
    else:
        raise ValueError("Invalid target mode. Choose 'multi_label' or 'one_hot'.")
    
    df = pd.read_csv(os_path.join(dataset_path, dataset_name), sep='\t')
    y = pd.DataFrame(transform_target(df['tags'].apply(literal_eval)))  # process the labels, apply literal_eval to convert strings to lists
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

    return xs_padded, ys

class MelspecsDataset(Dataset):
    """
    Loader of mel-spectrograms dataset.
    """
    def __init__(self, x, y, device, dataset_path, transform_specs=None):
        self.x = x
        self.y = y
        self.dataset_path = dataset_path
        self.transform_specs = transform_specs
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        row = self.x.iloc[idx]

        spec = np.load(os_path.join(self.dataset_path, row['melspecs_path']))
        spec = from_numpy(spec).float().to(self.device)  # size (num_mels, num_frames<=seq_len>)

        # Apply transformation to the mel spectrogram if specified
        if self.transform_specs:
            spec = self.transform_specs(spec)

        label = torch_tensor(self.y.iloc[idx], dtype=torch_float).to(self.device)  # size (num_classes,)
        return spec, label
