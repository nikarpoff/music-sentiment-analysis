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
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import scipy.stats

# Multithreading imports
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from argparse import ArgumentParser


def cli_arguments_preprocess() -> tuple:
    parser = ArgumentParser(description="Extract audio features from a dataset and saves result to path of source dataset.")

    parser.add_argument("--path", required=True,
                      help="Path to source dataset")
    
    parser.add_argument("--name", required=False, default="dataset_all_moods.tsv",
                      help="Source dataset filename")

    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    return os.path.abspath(args.path), args.name


def extract_features(waveform, sr=22050, hop_length=512, n_mfcc=13):
    """
    Extract audio features from a file using librosa and essentia.
    Returns a dict of aggregated frame-wise features.
    """
    # Load audio
    duration = librosa.get_duration(y=waveform, sr=sr)

    # Frame-wise features
    zcr = librosa.feature.zero_crossing_rate(waveform, hop_length=hop_length)[0]
    rmse = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=waveform, hop_length=hop_length)[0]
    flux = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=waveform, sr=sr)

    # Beat and rhythm via librosa
    tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onsets) / duration

    # Helper: aggregate stats
    def stats(x):
        return {
            'mean': np.mean(x), 'std': np.std(x),
            'skew': scipy.stats.skew(x), 'kurtosis': scipy.stats.kurtosis(x),
            'p25': np.percentile(x, 25), 'p50': np.percentile(x, 50), 'p75': np.percentile(x, 75)
        }

    features = {}
    # Time-domain
    features.update({f'zcr_{k}': v for k, v in stats(zcr).items()})
    features.update({f'rmse_{k}': v for k, v in stats(rmse).items()})

    # Spectral
    features.update({f'centroid_{k}': v for k, v in stats(centroid).items()})
    features.update({f'bandwidth_{k}': v for k, v in stats(bandwidth).items()})
    features.update({f'rolloff_{k}': v for k, v in stats(rolloff).items()})
    features.update({f'flatness_{k}': v for k, v in stats(flatness).items()})
    features.update({f'flux_{k}': v for k, v in stats(flux).items()})

    # Chroma and Tonnetz
    for i in range(chroma.shape[0]):
        features.update({f'chroma{i+1}_{k}': v for k, v in stats(chroma[i]).items()})
    for i in range(tonnetz.shape[0]):
        features.update({f'tonnetz{i+1}_{k}': v for k, v in stats(tonnetz[i]).items()})

    # Rhythm
    try:
        features['tempo_librosa'] = float(tempo[0]) if hasattr(tempo, '__getitem__') else float(tempo)
    except:
        features['tempo_librosa'] = 0.0

    features['onset_rate'] = onset_rate

    return features


def process_row(row_dict, dataset_path):
    try:
        path = os.path.join(dataset_path, row_dict["path"])
        waveform, sr = sf.read(path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)                # Convert to mono if stereo
        waveform = waveform[: min(len(waveform), 60 * sr)]  # Limit to 1 minute
        features = extract_features(waveform, sr)
        features["track_id"] = row_dict["track_id"]
        features["path"] = row_dict["path"]
        features["tags"] = row_dict["tags"]
        return features
    except Exception as e:
        print(f"Error processing {row_dict['path']}: {e}")
        return None


if __name__ == "__main__":
    dataset_path, dataset_source_name = cli_arguments_preprocess()
    dataset_full_path = os.path.join(dataset_path, dataset_source_name)
    save_dataset_path = os.path.join(dataset_path, f"features_{dataset_source_name}")

    # Read the dataset without splitting the last column.
    with open(dataset_full_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    columns_number = len(lines[0].split("\t"))

    # Split lines by tabs; last column may contain multiple values.
    data = [line.strip().split("\t", maxsplit=columns_number - 1) for line in lines]
    
    # Create a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])  # First row as header

    print("Start to extract features...")

    # Multithreading processing
    rows = df.to_dict(orient='records')
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    partial(
                        process_row, dataset_path=dataset_path
                    ), rows
                ), total=len(rows)
            )
        )
    results = [r for r in results if r is not None]
    print("Features are extracted from all tracks!")

    target_df = pd.DataFrame(results)
    target_df.to_csv(save_dataset_path, sep="\t", index=False)

    print(f"\t Target dataset saved to {save_dataset_path}")
