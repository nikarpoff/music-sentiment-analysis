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

from torch import device as torch_device
from torch.cuda import is_available as is_cuda_available
from dotenv import load_dotenv

from argparse import ArgumentParser
from model.data import load_specs_dataset
from model.specstr import SpectrogramTransformer
from model.utils import train_model, evaluate_model


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
                        choices=["specstr"],
                        help="Type of machine learning model to be used")
    
    parser.add_argument("--task", required=True,
                        choices=["train", "test"],
                        help="Type of task to be performed")
    
    parser.add_argument("--moods", required=False,
                      choices=["2", "4", "8", "all"],
                      help="Number of aggregated moods. By default: all source moods")
    
    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    if not args.moods or args.moods == "all":
        args.moods = 0

    return os.path.abspath(args.path), args.model, args.task, args.moods

if __name__ == "__main__":
    # Read command line arguments.
    dataset_path, model_type, task_type, moods_number = cli_arguments_preprocess()

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

    # Set the dataset name based on the moods number.
    dataset_name = "dataset_all_moods.tsv" if moods_number == 0 else f"dataset_{moods_number}_moods.tsv"
    output_dim = total_moods if moods_number == 0 else moods_number
    device = torch_device("cuda" if is_cuda_available() else "cpu")

    # Select the target transformation based on the target mode.
    if moods_number == 0:
        output_activation = "sigmoid"
        target_mode = "multi_label"
        loss = "binary_cross_entropy_with_logits"
    else:
        output_activation = "softmax"
        target_mode = "one_hot"
        loss = "cross_entropy"

    if model_type == "specstr":
        d_model = int(os.getenv("SPECSTR_D_MODEL", 512))
        nhead = int(os.getenv("SPECSTR_NHEAD", 32))
        num_layers = int(os.getenv("SPECSTR_NUM_LAYERS", 6))
        seq_len = int(os.getenv("SPECSTR_SEQ_LEN", 1024))

        # Load the dataset.
        train_loader, val_loader, test_loader = load_specs_dataset(dataset_path, dataset_name, device, target_mode, seq_len=seq_len,
                                                                   val_size=0.2, test_size=0.2, random_state=7)

        model = SpectrogramTransformer(d_model=d_model, output_dim=output_dim,
                                       nhead=nhead, num_layers=num_layers,
                                       output_activation=output_activation, device=device).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if task_type == "train":
        train_model(model, save_path, loss, train_loader, val_loader, lr=learning_rate, epochs=epochs, l2_reg=l2_reg)
    elif task_type == "test":
        pass
    else:
        raise ValueError(f"Unknown task type: {task_type}")
