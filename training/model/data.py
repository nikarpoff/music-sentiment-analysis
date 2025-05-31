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
import torchaudio
import json
import numpy as np
import pandas as pd
from collections.abc import Iterator
from abc import abstractmethod

from random import randint
from ast import literal_eval
from os import path as os_path
from datetime import datetime

import torchaudio.transforms as T
from torch_audiomentations import Compose, PitchShift, Shift
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder


from config import *


def one_hot_encode_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the labels.

    This encoder used for 2/4/8-moods dataset.
    """
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(labels)
    classes = encoder.categories_[0].tolist()
    return one_hot_labels, classes

def multi_label_binarize(labels: pd.Series) -> pd.Series:
    """
    Multi Label Binarization encode the labels.

    This encoder used for all-moods dataset.
    """
    encoder = MultiLabelBinarizer()
    binarized_labels = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()
    return binarized_labels, classes

def label_encode(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Simple label encoding.
    :return: encoded labels and initial classes list (for parsing to string back)
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()
    return encoded_labels, classes

def melspecs_classify_collate_fn(batch):
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

def melspecs_autoencode_collate_fn(batch):
    """
    Custom collaction function with y = spec output
    """
    xs, _ = zip(*batch)
    xs = [x.permute(1, 0) for x in xs] 

    # Padding by maximum seq_len.
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.)

    # Return initial dimensions order (batch, channel, time)
    xs_padded = xs_padded.permute(0, 2, 1)

    return xs_padded, xs_padded

class PadAugmentCollate:
    def __init__(self, pad_value: float, augmentation: TransformerMixin | None = None, for_train: bool = False):
        self.pad_value = pad_value
        self.augmentation = augmentation
        self.for_train = for_train

    def __call__(self, batch):
        # batch: list of (waveform: Tensor[T], label: Tensor)
        xs, ys = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=self.pad_value)  # (B, T_max)
        xs_padded = xs_padded.unsqueeze(1)  # add channel dim (B, 1, T_max)

        # Augmentation if required
        if self.for_train and (self.augmentation is not None):
            xs_padded = self.augmentation.transform(xs_padded)

        ys = torch.stack(ys)

        # for i in range(xs_padded.shape[0]):
        #     torchaudio.save(
        #         uri=f"F:/dataset/music/augmented{i}_{ys[i].argmax()}.wav",
        #         src=xs_padded[i],
        #         sample_rate=16000,
        #     )
        return xs_padded, ys


class KFoldDataLoader(Iterator):
    """
    Interface of DataLoader objects for train/val folds and testing supsets. Iterable by folds.
    """
    def __init__(self, dataset_path: str, dataset_name: str, splits: int, target_mode: str, test_size=0.2,
                 batch_size=32, num_workers=8, outputs_path='./outputs', moods="all", random_state=None):
        """
        :param dataset_path: Path to the dataset directory.
        :param dataset_name: Name of the dataset file (depends on moods mode).
        :param splits: Number of folds.
        :param test_size: Proportion of the dataset to include in the test split.
        :param batch_size: Batch size for the DataLoader.
        :param num_workers: Number of subprocesses to use for data loading.
        :param outputs_path: Path to save tags and labels conformity.
        :param random_state: Random seed for reproducibility.
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.full_dataset_path = os_path.join(dataset_path, dataset_name)
        self.random_state = random_state
        self.outputs_path = outputs_path
        self.num_workers = num_workers
        self.target_mode = target_mode
        self.batch_size = batch_size
        self.test_size = test_size
        self.splits = splits
        self.classes = None
        self.moods = moods

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
    
    def _load_train_test(self) -> tuple:
        """
        Service function for loading and spliting .tsv dataset
        """
        df = pd.read_csv(self.full_dataset_path, sep='\t')

        # Select the target transformation based on the target mode.
        if self.target_mode == MULTILABEL_TARGET:
            y, classes = multi_label_binarize(df['tags'].apply(literal_eval))  # process the labels, apply literal_eval to convert strings to lists
        elif self.target_mode == ONE_HOT_TARGET:
            y, classes = one_hot_encode_labels(df['tags'].to_frame())  # process the labels to one-hot vectors
        else:
            y, classes = label_encode(df['tags'])  # simple label encoding

        # Use dataframe next.
        y = pd.DataFrame(y)

        # Save classes and encoded labels conformity to json file.
        if self.random_state is not None:
            version = self.random_state
        else:
            version = datetime.now().strftime(DATE_FORMAT)

        self.classes = classes
        classes_filename = f"classes_{self.target_mode}_{self.moods}_{version}.json"
        with open(os_path.join(self.outputs_path, classes_filename), "w", encoding="utf-8") as file:
            json.dump(classes, file, indent=4)

        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=self.test_size, random_state=self.random_state)

        return x_train, x_test, y_train, y_test

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
        train_loader = self._get_loader(x_train, y_train, for_train=True)
        val_loader = self._get_loader(x_val, y_val, for_train=False)

        return train_loader, val_loader
    
    def get_train_len(self):
        return len(self.x_train)
    
    def set_start(self, start):
        """
        Required for loading from checkpoints.
        """
        self.start_fold = start

    @abstractmethod
    def _get_loader(self, x, y, for_train=True):
        """Get loader of batched data"""

    def get_test_loader(self):
        return self._get_loader(self.x_test, self.y_test, for_train=False), self.classes

    
    def __len__(self):
        return self.splits


class KFoldSpecsDataLoader(KFoldDataLoader):
    """
    Load the dataset and create DataLoader objects for train/val folds and testing supsets.

    Splits train/test as 1 - test_size / test_size (0.8 / 0.2 by default).

    Applyes transformation transform_specs for spectrogramms.
    Truncates/paddes spectrogramms if required (if min_seq_len or max_seq_len provided).
    """
    def __init__(self, dataset_path: str, dataset_name: str, splits: int, target_mode: str, min_seq_len: None, max_seq_len: None,
                pad_value=-90., test_size=0.2, batch_size=32, transform_specs: TransformerMixin = None, use_augmentation = True,
                num_workers=8, outputs_path='./outputs', moods="all", random_state=None):
        """
        :param max_seq_len: Constant max sequence length. Spectrograms with another length will be truncated to this length. Optional
        :param min_seq_len: Constant min sequence length. Spectrograms with another length will be padded to this length. Optional
        :param pad_value: Pad value for cases where padding of spectrogram required.
        :param transform_specs: sklearn transformer for preprocessing spectrograms. Optional.
        """
        super().__init__(dataset_path, dataset_name, splits, target_mode, test_size, batch_size, num_workers, outputs_path, moods, random_state)

        self.transform_specs = transform_specs
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

        self.augmentation = None
        if use_augmentation:
            self.augmentation = SpecAugment(mask_value=pad_value)

    def __iter__(self):
        return self

    def __next__(self):
        return super().__next__()

    def _get_loader(self, x, y, for_train=True):
        if self.target_mode == AUTOENCODER_TARGET:
            collate_fn = melspecs_autoencode_collate_fn
        else:
            collate_fn = melspecs_classify_collate_fn

        dataset = MelspecsDataset(x, y, dataset_path=self.dataset_path, transform_specs=self.transform_specs,
                                  augmentation=self.augmentation, max_seq_len=self.max_seq_len, training=for_train,
                                  min_seq_len=self.min_seq_len, pad_value=self.pad_value)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )


class KFoldRawAudioDataLoader(KFoldDataLoader):
    """
    Load the dataset and create DataLoader objects for train/val folds and testing supsets.

    Splits train/test as 1 - test_size / test_size (0.8 / 0.2 by default).

    Applyes transformation transform_specs for raw audiodata.
    Truncates/paddes audio if required (if min_seq_len or max_seq_len provided).
    """
    def __init__(self, dataset_path: str, dataset_name: str, splits: int, target_mode: str, min_seq_len: None, max_seq_len: None,
                pad_value=0., test_size=0.2, batch_size=32, transform_audio: TransformerMixin = None, use_augmentation = True,
                sample_rate=22050, num_workers=8, outputs_path='./outputs', moods="all", random_state=None):
        """
        :param max_seq_len: Constant max sequence length. Spectrograms with another length will be truncated to this length. Optional
        :param min_seq_len: Constant min sequence length. Spectrograms with another length will be padded to this length. Optional
        :param pad_value: Pad value for cases where padding of spectrogram required.
        :param transform_specs: sklearn transformer for preprocessing spectrograms. Optional.
        """
        super().__init__(dataset_path, dataset_name, splits, target_mode, test_size, batch_size, num_workers, outputs_path, moods, random_state)
        self.transform_audio = transform_audio
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.pad_value = pad_value

        self.augmentation = None
        if use_augmentation:
            self.augmentation = AudioAugment(sample_rate=sample_rate)

    def __iter__(self):
        return self

    def __next__(self):
        return super().__next__()

    def _get_loader(self, x, y, for_train=True):
        collate_fn = PadAugmentCollate(pad_value=self.pad_value, augmentation=self.augmentation, for_train=for_train)

        dataset = RawAudioDataset(x, y, dataset_path=self.dataset_path, sample_rate=self.sample_rate,
                                  max_seq_len=self.max_seq_len, min_seq_len=self.min_seq_len,
                                  pad_value=self.pad_value)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
        )
    

class RawAudioDataset(Dataset):
    """
    Loader of raw audio dataset.
    """
    def __init__(self, x, y, dataset_path, min_seq_len, max_seq_len, pad_value, sample_rate = 22050):
        self.x = x
        self.y = y
        self.dataset_path = dataset_path
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.sample_rate = sample_rate
        self.resampler = None

        self.start_skip_frames = sample_rate * 1  # skip a first second in audio
        self.end_skip_frames = sample_rate * 1  # skip a last second in audio

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        row = self.x.iloc[idx]
        audio_path = os_path.join(self.dataset_path, row['path'])
        
        # Number of input samples.
        segment_len = self.max_seq_len or self.sample_rate  # if not specified -> 1 second

        # Get total len of audio.
        info = torchaudio.info(audio_path)
        total_len = info.num_frames

        # Select random part of audio.
        if total_len > segment_len + self.end_skip_frames:
            start = torch.randint(self.start_skip_frames,
                total_len - segment_len - self.end_skip_frames + 1, (1,)
            ).item()
        else:
            start = self.start_skip_frames

        audio, sr = torchaudio.load(
            audio_path,
            frame_offset=start,
            num_frames=segment_len,
            normalize=True
        )

        # Resample if needed (often downsample)
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = self.resampler(audio)

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio.mean(dim=0)

        label = torch.tensor(self.y.iloc[idx], dtype=torch.float)  # size (num_classes,)
        return audio, label


class MelspecsDataset(Dataset):
    """
    Loader of mel-spectrograms dataset.
    """
    def __init__(self, x, y, dataset_path, min_seq_len, max_seq_len, pad_value, training = True,
                 transform_specs: TransformerMixin = None, augmentation: TransformerMixin = None):
        self.x = x
        self.y = y
        self.dataset_path = dataset_path
        self.training = training
        self.augmentation = augmentation
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
        
        # Apply augmentation. Use audio augmentation before scaling/normalization!
        if self.augmentation and self.training:
            spec = self.augmentation.transform(spec)

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


class SpecAugment(TransformerMixin):
    """
    Spectrograms augmentation. Performs random Gaussian noise, applies masking and gain with some probability.
    Works with unscaled spectrograms in decibels (dB).
    """
    def __init__(self, freq_mask_param=6, time_mask_param=64, mask_value=-90., p_f=0.3, p_t=0.3, p_g=0.3):
        self.p_f = p_f
        self.p_t = p_t
        self.p_g = p_g

        self.mask_value = mask_value
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.randomGain = RandomGain(min_gain=-5, max_gain=5)   # gain in db
        self.gaussianNoise = GaussianNoise(std=0.4)             # noise in db

    def transform(self, spec: np.ndarray) -> np.ndarray:
        spec = self.gaussianNoise(spec)

        # Apply gain with probability p_g.
        if np.random.rand() < self.p_g:
            spec = self.randomGain(spec)

        spec_tensor = torch.from_numpy(spec).float()

        # Apply frequency masking with probability p_f.
        if np.random.rand() < self.p_f:
            spec_tensor = self.freq_mask(spec_tensor, mask_value=self.mask_value)
        
        # Apply time masking with probability p_t.
        if np.random.rand() < self.p_t:
            spec_tensor = self.time_mask(spec_tensor, mask_value=self.mask_value)
        
        return spec_tensor.numpy()


class AudioAugment(TransformerMixin):
    def __init__(self, sample_rate, p_f=0.3, p_g=0.3, p_e=0.3):
        """
        Batch-level augmentation (expects batch x channels x time)
        """
        self.sample_rate = sample_rate
        self.p_f = p_f
        self.p_g = p_g
        self.p_e = p_e

        self.fade = T.Fade(fade_in_len=int(0.1 * sample_rate), fade_out_len=int(0.1 * sample_rate))  # 10% fade in/out

    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random gain
        if torch.rand(1).item() < self.p_g:
            # Random gain factor between 0.6 and 1.1
            gain_factor = torch.empty(1).uniform_(0.6, 1.1).item()
            vol = T.Vol(gain_factor, gain_type="amplitude")
            waveform = vol(waveform)

        # Random effects
        waveform = self.apply_random_effects(waveform)

        # Fade in/out
        if torch.rand(1).item() < self.p_f:
            waveform = self.fade(waveform)

        # Gaussian noise
        waveform = self.apply_noise(waveform)

        return waveform

    def apply_random_effects(self, waveform):
        augment = Compose(
            transforms=[
                PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, p=self.p_e, sample_rate=self.sample_rate, output_type='tensor'),
                # Shift(min_shift=-1000, max_shift=1000, p=self.p_e, sample_rate=self.sample_rate, output_type='tensor'),
            ],
            output_type='tensor'
        )

        return augment(waveform, sample_rate=self.sample_rate)

    def apply_noise(self, waveform):
        max_amp = 0.001
        random_noise_amp = torch.randn(1).item() * max_amp
        noise = torch.randn_like(waveform) * random_noise_amp
        return waveform + noise

    def apply_gain(self, waveform):
        gain = torch.empty(1).uniform_(-0.1, 0.1).item()
        return waveform + gain

class RandomGain():
    """
    Applyes random gain to the spectrogram.
    """
    def __init__(self, min_gain, max_gain):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        gain = np.random.uniform(self.min_gain, self.max_gain)
        return spec + gain
    
class GaussianNoise():
    """
    Applyes random noise to the data.
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(self.mean, self.std, data.shape)
        return data + noise
    