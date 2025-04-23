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
import numpy as np
import pandas as pd

import librosa
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def cli_arguments_preprocess() -> str:
    """
    Read, parse and preprocess command line arguments:
        - path to source dataset. Required
    """
    parser = ArgumentParser(description="Demo script for audio processing")

    parser.add_argument("--path", required=True,
                      help="Path to source dataset")
    
    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    return os.path.abspath(args.path)

# THIS DEMO REQUIES THE FFMPEG LIBRARY TO BE INSTALLED
def show_random_audio():
    dataset_path = cli_arguments_preprocess()
    cleaned_dataset_name = "dataset_all_moods.tsv"
    cleaned_dataset_path = os.path.join(dataset_path, cleaned_dataset_name)

    if not os.path.isfile(os.path.join(dataset_path, cleaned_dataset_path)):
        print(f"This script requires the {cleaned_dataset_name} file to be present. Please, run 'scriprs.preprocess_dataset.py --path <dataset_path>' first.")
        return

    dataset = pd.read_csv(cleaned_dataset_path, sep="\t")

    # Show random audio amplitudes and spectrograms.
    random_row = dataset.sample(n=1)
    random_song_path = os.path.abspath(os.path.join(dataset_path, random_row.iloc[0]["path"]))
    random_spec_path = os.path.abspath(os.path.join(dataset_path, random_row.iloc[0]["melspecs_path"]))
    print(f"Random song selected:\n{random_song_path}\n")

    if not os.path.isfile(random_song_path):
        print(f"File {random_song_path} not found. Please, ensure that you have the dataset and the path is correct.")
        return

    amplitudes, sample_rate = librosa.core.load(random_song_path)
    print(f"Audio length: {amplitudes.shape[0] / sample_rate:.3f} seconds at sample rate {sample_rate}")
    print(f"Samples number {amplitudes.shape[0]}\n")

    # Generate a spectrogram using Short-Time Fourier Transform (STFT)
    window_length = 2048
    hop_length = 512
    window_time = window_length / sample_rate

    spectrogram = librosa.stft(amplitudes, n_fft=window_length, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    print(f"STFT-built spectrogram shape: {spectrogram_db.shape}")

    # Load Mel spectrogram
    mel_spec = np.load(random_spec_path)
    print(f"Dataset mel-spectrogram shape: {mel_spec.shape}")

    show_audio_time = 10  # demo seconds
    show_samples_number = show_audio_time * sample_rate
    show_windows_number = int(show_audio_time / window_time)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [3, 1, 1]})
    ax1.set_title(f"First {show_audio_time} seconds of the audio")
    ax1.plot(amplitudes[:show_samples_number])
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")

    ax2.set_title(f"Spectrogram ({show_audio_time}s)")
    ax2.imshow(spectrogram_db[:, :show_windows_number], aspect='auto', origin='lower', cmap='plasma', extent=[0, show_windows_number, 0, spectrogram_db.shape[0]])
    ax2.set_xlabel("Window")
    ax2.set_ylabel("Frequency (Hz)")
    
    ax3.set_title(f"Mel-Spectrogram ({show_audio_time}s)")
    ax3.imshow(mel_spec[:, :show_windows_number], aspect='auto', origin='lower', cmap='plasma', extent=[0, show_windows_number, 0, mel_spec.shape[0]])
    ax3.set_xlabel("Window")
    ax3.set_ylabel("Mel frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_random_audio()
