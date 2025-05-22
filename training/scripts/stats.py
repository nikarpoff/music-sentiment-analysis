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
import json
import numpy as np
import pandas as pd
from scipy import stats as st
from dotenv import load_dotenv

from argparse import ArgumentParser


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


def get_specs_stats():
    """
    Collects statistics about mel-specs and dumps them to the melspecs_stats.json.
    """
    dataset_path = cli_arguments_preprocess()
    cleaned_dataset_name = "dataset_all_moods.tsv"
    cleaned_dataset_path = os.path.join(dataset_path, cleaned_dataset_name)

    if not os.path.isfile(os.path.join(dataset_path, cleaned_dataset_path)):
        print(f"This script requires the {cleaned_dataset_name} file to be present. Please, run 'scriprs.preprocess_dataset.py --path <dataset_path>' first.")
        return

    load_dotenv()
    outputs_path = os.getenv("OUTPUTS_PATH")

    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    dataset = pd.read_csv(cleaned_dataset_path, sep="\t")

    # Charcteristics of specs.
    n_audios = len(dataset)
    total_analyzed = n_audios
    samples_counts = []
    min_amplitudes = []
    max_amplitudes = []
    
    print(f"Start getting statistics for {n_audios} audios")

    # Get random N_AUDIOS audio spectrograms and count characteristics.
    for i in range(n_audios):
        melspec_path = os.path.join(dataset_path, dataset.loc[i, "melspecs_path"])
        
        if not os.path.isfile(melspec_path):
            print(f"Warning: {melspec_path} not found! This file will be skiped!")
            total_analyzed -= 1
            continue
        
        melspec = np.load(melspec_path)

        min_amplitudes.append(np.min(melspec))
        max_amplitudes.append(np.max(melspec))
        samples_counts.append(melspec.shape[1])

    avg_samples_count = np.mean(samples_counts)
    min_samples_count = np.min(samples_counts)
    max_samples_count = np.max(samples_counts)

    max_amplitude = np.max(max_amplitudes)
    mean_max_amplitude = np.mean(max_amplitudes)
    mode_max_amplitude = st.mode(max_amplitudes).mode

    min_amplitude = np.min(min_amplitudes)
    mean_min_amplitude = np.mean(min_amplitudes)
    mode_min_amplitude = st.mode(min_amplitudes).mode

    # Display audio statistics
    print(f"Audio statistics (analyzed {total_analyzed} files):")
    print(f"\n\t Average samples count: {avg_samples_count:.2f}")
    print(f"\t Max samples count: {max_samples_count}")
    print(f"\t Min samples count: {min_samples_count}")

    print(f"\n\t Max amplitude (dB): {max_amplitude:.2f}")
    print(f"\t Mean-Max amplitude (dB): {mean_max_amplitude:.2f}")
    print(f"\t Mode-Max amplitude (dB): {mode_max_amplitude:.2f}")

    print(f"\n\t Min amplitude (dB): {min_amplitude:.2f}")
    print(f"\t Mean-Min amplitude (dB): {mean_min_amplitude:.2f}")
    print(f"\t Mode-Min amplitude (dB): {mode_min_amplitude:.2f}")

    stats = {
        "total_melspecs": int(total_analyzed),
        "average_samples_count": float(avg_samples_count),
        "max_samples_count": int(max_samples_count),
        "min_samples_count": int(min_samples_count),
        "max_amplitude": float(max_amplitude),
        "mean_max_amplitude": float(mean_max_amplitude),
        "mode_max_amplitude": float(mode_max_amplitude),
        "min_amplitude": float(min_amplitude),
        "mean_min_amplitude": float(mean_min_amplitude),
        "mode_min_amplitude": float(mode_min_amplitude),
    }

    with open(os.path.join(outputs_path, "melspecs_stats.json"), "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

if __name__ == "__main__":
    get_specs_stats()
