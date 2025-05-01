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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import json
import torch
import numpy as np
from dotenv import load_dotenv

from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from model.data import KFoldSpecsDataLoader
from model.specstr import SpectrogramTransformer, SpectrogramPureTransformer
from model.utils import ClassificationModelTrainer, evaluate_model


def cli_arguments_preprocess() -> str:
    """
    Read, parse and preprocess command line arguments:
        - path to source dataset. Required
        - moods number. By default: all source moods
        - task type. Required
        - model type. Required
    """
    parser = ArgumentParser(description="Main script for training and inference audio models.")

    parser.add_argument("--path", required=True,
                        help="Path to source dataset")
    
    parser.add_argument("--model", required=True,
                        choices=["pure_specstr", "prelearned_specstr", "specstr"],
                        help="Type of machine learning model to be used")
    
    parser.add_argument("--task", required=True,
                        choices=["train", "ctrain", "test"],
                        help="Type of task to be performed")
    
    parser.add_argument("mname", nargs="?", help="Model name for load and test/continue train (Required for task == test or ctrain)")
    
    parser.add_argument("--moods", required=False,
                      choices=["2", "4", "8", "all"],
                      help="Number of aggregated moods. By default: all source moods")
    
    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    if not args.moods or args.moods == "all":
        args.moods = 0

    if args.mname is None and (args.task == "test" or args.task == "ctrain"):
        parser.error("For load model you must provide the name of saved model (with file extention)")

    return os.path.abspath(args.path), args.model, args.mname, args.task, int(args.moods)

def get_specs_scaler(melspecs_stats_path) -> MinMaxScaler:
    """
    Loads MinMaxScaler for mel-spectrograms transform
    """
    if os.path.isfile(melspecs_stats_path):
        with open(melspecs_stats_path) as file:
            stats = json.load(file)

            min_amplitude = stats["max_amplitude"]
            max_amplitude = stats["min_amplitude"]
    else:
        print("Warning! For MinMaxScaler max/min amplitudes from melspecs_stats.json required! Now using min/max defaults values.")
        print("You can run scripts.stats to get required statistics and avoid this warning.")
        
        min_amplitude = -90.
        max_amplitude = 29.6

    # Average audio can have -90 min amplitude and +29.6 max amplitude. So melspecs should be scaled. 
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Cause fitting MinMaxScaler is too expencieve, params can be set manually
    scaler.min_ = np.array([0])
    scaler.scale_ = np.array([1 / (max_amplitude - min_amplitude)])
    scaler.data_min_ = np.array([min_amplitude])
    scaler.data_max_ = np.array([max_amplitude])
    scaler.data_range_ = np.array([max_amplitude - min_amplitude])

    return scaler, min_amplitude, max_amplitude


if __name__ == "__main__":
    # Read command line arguments.
    dataset_path, model_type, model_name, task_type, moods_number = cli_arguments_preprocess()

    # Load environment variables.
    load_dotenv()
    outputs_path = str(os.getenv("OUTPUTS_PATH", "./outputs/"))
    models_path = str(os.getenv("MODELS_PATH", "./outputs/models/"))
    save_path = str(os.getenv(f"{model_type.upper()}_SAVE_PATH", "./outputs/models/"))
    
    if not os.path.isdir(outputs_path):
        os.mkdir(outputs_path)
    
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    learning_rate = float(os.getenv(f"{model_type.upper()}_LEARNING_RATE", 0.001))
    batch_size = int(os.getenv(f"{model_type.upper()}_BATCH_SIZE", 32))
    epochs = int(os.getenv(f"{model_type.upper()}_EPOCHS", 10))
    l2_reg = float(os.getenv(f"{model_type.upper()}_L2_REG", 0.01))
    total_moods = int(os.getenv("TOTAL_MOODS", 59))
    kfold_splits = int(os.getenv("KFOLD_SPLITS", 5))

    # Set the dataset name based on the moods number.
    dataset_name = "dataset_all_moods.tsv" if moods_number == 0 else f"dataset_{moods_number}_moods.tsv"
    output_dim = total_moods if moods_number == 0 else moods_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select the target transformation based on the target mode.
    if moods_number == 0:
        target_mode = "multilabel"
    else:
        target_mode = "onehot"

    # Load required dataset.
    if model_type == "pure_specstr" or model_type == "prelearned_specstr" or model_type == "specstr":
        max_seq_len = int(os.getenv(f"{model_type.upper()}_MAX_SEQ_LEN", 25000))
        min_seq_len = int(os.getenv(f"{model_type.upper()}_MIN_SEQ_LEN", 1000))

        # For MinMaxScaler max/min amplitudes required.
        transform_specs, min_amp, _ = get_specs_scaler(os.path.join(outputs_path, "melspecs_stats.json"))

        # Load the dataset.
        kfold_dataloader = KFoldSpecsDataLoader(dataset_path, dataset_name, kfold_splits, target_mode, pad_value=min_amp,
                                                batch_size=batch_size, max_seq_len=max_seq_len, min_seq_len=min_seq_len, 
                                                num_workers=4, test_size=0.2, transform_specs=transform_specs, random_state=7)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "specstr":
        dropout = float(os.getenv("SPECSTR_DROPOUT", 0.2))

        model = SpectrogramTransformer(output_dim=output_dim, dropout=dropout, device=device).to(device)
    elif model_type == "pure_specstr":
        model = SpectrogramPureTransformer(output_dim=output_dim, device=device).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if task_type == "train" or task_type == "ctrain":
        trainer = ClassificationModelTrainer(
            model, model_name=model_type, num_classes=output_dim, save_path=save_path,
            target_mode=target_mode, kfold_loader=kfold_dataloader, lr=learning_rate,
            epochs=epochs, l2_reg=l2_reg
        )

    if task_type == "train":
        # In train we starts with new model.
        print(f"Built model:\n", model)
        trainer.init_new_train()
        trainer.train_model()
    elif task_type == "test":
        # In test we use trained model.
        model.load_state_dict(torch.load(os.path.join(save_path, model_name), weights_only=True))
        print(f"Loaded model:\n", model)

        test_loader = kfold_dataloader.get_test_loader()

        evaluate_model(model, num_classes=output_dim, target_mode=target_mode, test_loader=test_loader)
    elif task_type == "ctrain":
        trainer.init_continue_train(saved_model_name=model_name)
        trainer.train_model()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
