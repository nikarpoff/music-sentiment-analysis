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

from config import *

from utils.utils import *
from utils.data import *
from model.text import *
from model.specstr import *
from model.rawtr import *
from model.features import *

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
    
    parser.add_argument("--dname", required=False,
                        help="Dataset name to load. By default: dataset_<moods>_moods.tsv")
    
    parser.add_argument("--model", required=True,
                        choices=AVAILABLE_MODELS,
                        help="Type of machine learning model to be used")
    
    parser.add_argument("--task", required=True,
                        choices=["train", "ctrain", "test"],
                        help="Type of task to be performed")
    
    parser.add_argument("mname", nargs="?", help="Model name to load and test/continue train or to use as pretrained part")
    
    parser.add_argument("--moods", required=False,
                      choices=["2", "hs", "re", "4", "8", "all"],
                      help="Number of aggregated moods or moods themselves. By default: all source moods")
    
    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    if not args.moods:
        args.moods = "all"

    if args.mname is None and (args.task == "test" or args.task == "ctrain"):
        parser.error("For load model you must provide the name of saved model (with file extention)")
    elif args.mname is None and args.model in PRETRAINED_MODELS:
        parser.error("For use pretrained model you must provide the name of saved model (with file extention)")

    if args.dname is None and args.model in FEATURES_MODELS:
        parser.error("For use any model based on features instead of audio/spectrograms you must provide dataset filename with features (dataset_features.tsv)")
    
    if args.dname is None:
        args.dname = "dataset_all_moods.tsv" if args.moods == "all" else f"dataset_{args.moods}_moods.tsv"

    return os.path.abspath(args.path), args.dname, args.model, args.mname, args.task, args.moods

def get_specs_scaler(melspecs_stats_path) -> MinMaxScaler:
    """
    Loads MinMaxScaler for mel-spectrograms transform
    """
    if os.path.isfile(melspecs_stats_path):
        with open(melspecs_stats_path) as file:
            stats = json.load(file)

            min_amplitude = stats["min_amplitude"]
            max_amplitude = stats["max_amplitude"]
    else:
        print("Warning! For MinMaxScaler max/min amplitudes from melspecs_stats.json required! Now using min/max defaults values.")
        print("You can run scripts.stats to get required statistics and avoid this warning.")
        
        min_amplitude = -90.
        max_amplitude = 29.6

    # Average audio can have -90 min amplitude and +29.6 max amplitude. So melspecs should be scaled. 
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Cause fitting MinMaxScaler is too expencieve, params can be set manually
    data_min  = np.array([min_amplitude])
    data_max  = np.array([max_amplitude])
    data_range = data_max - data_min
    scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / data_range
    min_   = scaler.feature_range[0] - data_min * scale_

    scaler.scale_ = scale_
    scaler.min_ = min_
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_range

    return scaler, min_amplitude, max_amplitude


if __name__ == "__main__":
    # Read command line arguments.
    dataset_path, dataset_name, model_type, model_name, task_type, moods = cli_arguments_preprocess()

    # Load environment variables.
    load_dotenv()
    outputs_path = str(os.getenv("OUTPUTS_PATH", "./outputs/"))
    models_path = str(os.getenv("MODELS_PATH", "./outputs/models/"))
    save_path = str(os.getenv(f"{model_type.upper()}_SAVE_PATH", "./outputs/models/"))
    random_state = str(os.getenv("RANDOM_STATE", "None"))
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

    learning_rate = float(os.getenv(f"{model_type.upper()}_LEARNING_RATE", 0.001))
    batch_size = int(os.getenv(f"{model_type.upper()}_BATCH_SIZE", 32))
    epochs = int(os.getenv(f"{model_type.upper()}_EPOCHS", 10))
    l2_reg = float(os.getenv(f"{model_type.upper()}_L2_REG", 0.01))
    total_moods = int(os.getenv("TOTAL_MOODS", 59))
    kfold_splits = int(os.getenv("KFOLD_SPLITS", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moods_number = total_moods

    if moods == "re" or moods == "hs":  # named moods distribution.
        moods_number = 2
    else:
        moods_number = int(moods_number)
    output_dim = moods_number

    # Select the target transformation based on the target mode.
    if model_type in AUTOENCODER_MODELS:
        target_mode = AUTOENCODER_TARGET
    elif model_type in CLASSIFICATION_MODELS and moods == "all":
        target_mode = MULTILABEL_TARGET
    else:
        target_mode = ONE_HOT_TARGET

    # Load required dataset.
    if model_type in SPECS_MODELS:
        max_seq_len = int(os.getenv(f"{model_type.upper()}_MAX_SEQ_LEN", 25000))
        min_seq_len = int(os.getenv(f"{model_type.upper()}_MIN_SEQ_LEN", 1000))

        # For MinMaxScaler max/min amplitudes required.
        transform_specs, min_amp, _ = get_specs_scaler(os.path.join(outputs_path, "melspecs_stats.json"))

        # Load the dataset.
        kfold_dataloader = KFoldSpecsDataLoader(dataset_path, dataset_name, kfold_splits, target_mode, pad_value=min_amp,
                                                batch_size=batch_size, max_seq_len=max_seq_len, min_seq_len=min_seq_len, 
                                                use_augmentation=True, num_workers=8, test_size=0.2, outputs_path=outputs_path,
                                                transform_specs=transform_specs, moods=moods, random_state=random_state)
    elif model_type in RAW_AUDIO_MODELS or model_type in LIRYCS_MODELS:  # raw audio models or audio->lirycs models required raw audio data loader
        max_seq_len = int(os.getenv(f"{model_type.upper()}_MAX_SEQ_LEN", 50000))
        min_seq_len = int(os.getenv(f"{model_type.upper()}_MIN_SEQ_LEN", 20000))
        sample_rate = int(os.getenv(f"{model_type.upper()}_SAMPLE_RATE", 22050))

        # Load the dataset. Normalization/scaling of audio not required.
        kfold_dataloader = KFoldRawAudioDataLoader(dataset_path, dataset_name, kfold_splits, target_mode, pad_value=-1.,
                                                batch_size=batch_size, max_seq_len=max_seq_len, min_seq_len=min_seq_len, 
                                                use_augmentation=True, num_workers=6, test_size=0.2, outputs_path=outputs_path,
                                                transform_audio=None, sample_rate=sample_rate, moods=moods, random_state=random_state)
    elif model_type in FEATURES_MODELS:
        kfold_dataloader = KFoldFeaturesDataLoader(dataset_path, dataset_name, kfold_splits, target_mode,
                                                   batch_size=batch_size, num_workers=2,
                                                   test_size=0.2, outputs_path=outputs_path, moods=moods,
                                                   random_state=random_state)
        input_features_number = kfold_dataloader.get_input_features_number()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build required model.
    if model_type == SPECSTR:
        dropout = float(os.getenv("SPECSTR_DROPOUT", 0.2))

        model = SpectrogramSmallTransformer(output_dim=output_dim, dropout=dropout, device=device).to(device)
    elif model_type == SPECS_AUTOENCODER:
        dropout = float(os.getenv("SPECS_AUTOENCODER_DROPOUT", 0.2))

        model = SpectrogramMaskedAutoEncoder(dropout=dropout, device=device).to(device)
    elif model_type == PURE_SPECSTR:
        model = SpectrogramPureTransformer(output_dim=output_dim, device=device).to(device)
    elif model_type == PRETRAINED_SPECSTR:
        dropout = float(os.getenv("PRETRAINED_SPECSTR_DROPOUT", 0.2))
        mae = SpectrogramMaskedAutoEncoder(dropout=dropout, device=device).to(device)

        # We can load autoencoder and train pre-trained model or load checkpoint of pre-trained model.
        if task_type == "train":
            # Load pretrained autoencoder (start train with pretrained autoencoder).
            load_path = os.path.join(os.getenv("SPECS_AUTOENCODER_SAVE_PATH"), model_name)
            mae.load_state_dict(torch.load(load_path, weights_only=True))
        # In other case, we start with pretrained model checkpoint, loading encoder params not required.

        encoder = mae.encoder
        model = SpectrogramPreTrainedTransformer(encoder, output_dim, device=device).to(device)
    elif model_type == RAWTR:
        dropout = float(os.getenv("RAWTR_DROPOUT", 0.2))
        model = RawAudioTransformer(output_channels=output_dim, dropout=dropout, device=device).to(device)
    elif model_type == PRETRAINED_RAWTR:
        dropout = float(os.getenv("RAWTR_DROPOUT", 0.2))
        model = PretrainedRawAudioTransformer(output_channels=output_dim, dropout=dropout, device=device).to(device)
    elif model_type == FEAMLP:
        dropout = float(os.getenv("FEAMLP_DROPOUT", 0.1))
        model = FeaturesDense(input_features_number ,output_channels=output_dim, dropout=dropout, device=device).to(device)        
    elif model_type == LIRYCSTR:
        dropout = float(os.getenv("LIRYCSTR_DROPOUT", 0.2))
        max_text_len = int(os.getenv("LIRYCSTR_MAX_TEXT_LEN", 128))
        whisper_model_name = str(os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small"))

        text_extractor = TextExtractor(max_seq_len=max_text_len, whisper_model_name=whisper_model_name, device=device).to(device)
        text_transformer = TextTransformer(depth=256, nheads=8, num_encoders=8, dropout=dropout, whisper_model_name=whisper_model_name, device=device).to(device)
        model = LirycsSentimentTransformer(text_extractor, text_transformer, output_channels=output_dim, device=device).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Built model:\n", model)
    total_params, trainable_params = get_params_count(model)
    formated_total_params = format(total_params, ",").replace(",", " ")
    formated_trainable_params = format(trainable_params, ",").replace(",", " ")
    print(f"Total params: {formated_total_params}")
    print(f"Trainable params: {formated_trainable_params}\n")

    if task_type == "train" or task_type == "ctrain":
        if model_type in CLASSIFICATION_MODELS:
            trainer = ClassificationModelTrainer(
                model, model_name=model_type, num_classes=output_dim, save_path=save_path,
                target_mode=target_mode, kfold_loader=kfold_dataloader, lr=learning_rate,
                epochs=epochs, l2_reg=l2_reg
            )
        elif model_type in AUTOENCODER_MODELS:
            trainer = AutoencoderModelTrainer(
                model, model_name=model_type, save_path=save_path,
                kfold_loader=kfold_dataloader, lr=learning_rate,
                epochs=epochs, l2_reg=l2_reg
            )

        if task_type == "train":
            # In train we starts with new model.
            trainer.init_new_train()
        else:
            # In continues train we starts with loaded model.
            trainer.init_continue_train(saved_model_name=model_name)
        
        trainer.train_model()
    elif task_type == "test":
        # In test we use trained model.
        model.load_state_dict(torch.load(os.path.join(save_path, model_name), weights_only=True))
        test_loader, classes = kfold_dataloader.get_test_loader()

        if model_type in CLASSIFICATION_MODELS:
            evaluate_classification_model(model, classes, target_mode=target_mode, test_loader=test_loader)
        elif model_type in AUTOENCODER_MODELS:
            evaluate_autoencoder(model, test_loader)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
