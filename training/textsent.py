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

from argparse import ArgumentParser

import torch
from dotenv import load_dotenv

from model.data import *
from model.text import *
from model.utils import train_text_sentiment_model, get_params_count


def cli_arguments_preprocess() -> str:
    """
    Read, parse and preprocess command line arguments:
        - task type. Required
    """
    parser = ArgumentParser(description="Main script for training and inference audio models.")
        
    parser.add_argument("--task", required=True,
                        choices=["train", "test"],
                        help="Type of task to be performed")
    
    args = parser.parse_args()

    return args.task


if __name__ == "__main__":
    task_type = cli_arguments_preprocess()
    load_dotenv()

    outputs_path = str(os.getenv("OUTPUTS_PATH", "./outputs/"))
    models_path = str(os.getenv("MODELS_PATH", "./outputs/models/"))
    save_path = str(os.getenv("TEXTR_SAVE_PATH", "./outputs/models/"))
    random_state = str(os.getenv("RANDOM_STATE", "None"))

    whisper_model_name = str(os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small"))

    if random_state == "None":
        random_state = None
    else:
        random_state = int(random_state)
    
    if not os.path.isdir(outputs_path):
        os.mkdir(outputs_path)
    
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    learning_rate = float(os.getenv("TEXTR_LEARNING_RATE", 0.001))
    batch_size = int(os.getenv("TEXTR_BATCH_SIZE", 32))
    epochs = int(os.getenv("TEXTR_EPOCHS", 10))
    l2_reg = float(os.getenv("TEXTR_L2_REG", 0.01))
    max_text_len = int(os.getenv("TEXTR_MAX_TEXT_LEN", 128))
    dropout = float(os.getenv("TEXTR_DROPOUT", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = TextSentimentDataLoader(batch_size=batch_size,
                                         num_workers=8,
                                         max_length=max_text_len,
                                         whisper_model_name=whisper_model_name,
                                         random_state=random_state)
    train_loader, val_loader, test_loader = dataloader.get_loaders()

    model = TextSentimentTransformer(output_channels=3, dropout=dropout, whisper_model_name=whisper_model_name, device=device).to(device)

    print(f"Built model:\n", model)
    total_params, trainable_params = get_params_count(model)
    formated_total_params = format(total_params, ",").replace(",", " ")
    formated_trainable_params = format(trainable_params, ",").replace(",", " ")
    print(f"Total params: {formated_total_params}")
    print(f"Trainable params: {formated_trainable_params}\n")

    if task_type == "train":
        train_text_sentiment_model(
            model,
            model_name="textr",
            save_path=outputs_path,
            num_classes=3,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=learning_rate,
            epochs=epochs,
            l2_reg=l2_reg,
        )
    else:
        raise NotImplementedError("Test task is not implemented yet.")
    