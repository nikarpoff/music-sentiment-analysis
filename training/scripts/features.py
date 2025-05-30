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
import scipy.stats

from argparse import ArgumentParser


def cli_arguments_preprocess() -> tuple:
    """
    Read, parse and preprocess command line arguments:
        - path to source dataset. Required
        - path to mel spectrograms. Should be related to dataset path. By default: "melspecs/"
        - number of moods. By default: all moods (0)

    Sorce dataset file should be named "autotagging_moodtheme.tsv"
    """
    parser = ArgumentParser(description="Preprocessing autotagging moods dataset script. Cleans moods, aggregates them and saves preprocessed .tsv file")

    parser.add_argument("--path", required=True,
                      help="Path to source dataset")
    
    parser.add_argument("--name", required=True,
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
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, hop_length=hop_length)
    flux = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
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
    # contrast has shape (n_bands, frames)
    for i, band in enumerate(contrast):
        features.update({f'contrast_band{i+1}_{k}': v for k, v in stats(band).items()})
    features.update({f'flux_{k}': v for k, v in stats(flux).items()})

    # MFCCs
    for i in range(mfcc.shape[0]):
        features.update({f'mfcc{i+1}_{k}': v for k, v in stats(mfcc[i]).items()})
        features.update({f'delta_mfcc{i+1}_{k}': v for k, v in stats(delta_mfcc[i]).items()})
        features.update({f'delta2_mfcc{i+1}_{k}': v for k, v in stats(delta2_mfcc[i]).items()})

    # Chroma and Tonnetz
    for i in range(chroma.shape[0]):
        features.update({f'chroma{i+1}_{k}': v for k, v in stats(chroma[i]).items()})
    for i in range(tonnetz.shape[0]):
        features.update({f'tonnetz{i+1}_{k}': v for k, v in stats(tonnetz[i]).items()})

    # Rhythm
    features['tempo_librosa'] = tempo[0]
    features['onset_rate'] = onset_rate

    return features


if __name__ == "__main__":
    dataset_path, dataset_source_name = cli_arguments_preprocess()
    dataset_full_path = os.path.join(dataset_path, dataset_source_name)
    save_dataset_path = os.path.join(dataset_path, f"features_{dataset_source_name}")

    df = pd.read_csv(dataset_full_path, sep='\t')
    features_array = []

    total_len = len(df)
    report_times_percent = 10
    report_frequency = total_len // report_times_percent

    print("Start to extract features...")
    for i, row in df.iterrows():
        waveform, sr = librosa.load(os.path.join(dataset_path, row["path"]))
        features = extract_features(waveform, sr)
        features["track_id"] = row["track_id"]
        features["path"] = row["path"]
        features["tags"] = row["tags"]

        features_array.append(features)

        if i+1 % report_frequency == 0:
            print(f"\t Extracted {i+1}/{total_len}")

    print("Features are extracted from all tracks!")

    target_df = pd.DataFrame(features_array)
    target_df.to_csv(save_dataset_path, sep="\t", index=False)

    print(f"\t Target dataset saved to {save_dataset_path}")
