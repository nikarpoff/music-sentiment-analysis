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
from sklearn.preprocessing import MinMaxScaler

import torch
from dotenv import load_dotenv

from utils.data import *
from utils.utils import *

from model.specstr import *
from model.text import *
from model.rawtr import *
from model.features import *
from model.text import *
from model.full import *

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*MPEG_LAYER_III subtype is unknown to TorchAudio.*",
    category=UserWarning,
    module=r"torchaudio\._backend\.soundfile_backend"
)

def cli_arguments_preprocess() -> str:
    """
    Read, parse and preprocess command line arguments:
        - task type. Required
    """
    parser = ArgumentParser(description="Main script for training and inference audio models.")

    parser.add_argument("--path", required=True,
                        help="Path to the dataset")
    
    parser.add_argument("--task", required=True,
                        choices=["train", "ctrain", "test"],
                        help="Type of task to be performed")
    
    parser.add_argument("--moods", required=True,
                        choices=["hs", "re"],
                        help="Moods for train/test")
    
    parser.add_argument("--specstr", required=True,
                        help="Model name of specs transformer to be used for training")
    
    parser.add_argument("--rawtr", required=False,
                        help="Model name of raw audio transformer to be used for training")
    
    parser.add_argument("--textr", required=False,
                        help="Model name of text extraction and classification")
    
    parser.add_argument("--dname", required=False,
                        help="Dataset name to load. By default: dataset_<moods>_moods.tsv")
    
    parser.add_argument("--mname", required=False,
                        help="Model name to test/continue train")

    args = parser.parse_args()

    return args.path, args.task, args.moods, args.specstr, args.rawtr, args.textr, args.mname

def get_specs_scaler() -> MinMaxScaler:
    """
    Loads MinMaxScaler for mel-spectrograms transform
    """
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
    dataset_path, task_type, moods, specstr_name, rawtr_name, textr_name, model_name = cli_arguments_preprocess()
    load_dotenv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dropout = float(os.getenv("FULL_DROPOUT", 0.2))

    outputs_path = str(os.getenv("OUTPUTS_PATH", "./outputs/"))
    models_path = str(os.getenv("MODELS_PATH", "./outputs/models/"))
    save_path = str(os.getenv("FULL_SAVE_PATH", "./outputs/models/"))
    random_state = str(os.getenv("RANDOM_STATE", "None"))

    specstr_path = str(os.getenv("SPECSTR_SAVE_PATH", "./outputs/"))
    if "8M" in specstr_name:
        specstr = SpectrogramSmallTransformer(output_dim=2, dropout=dropout, device=device).to(device)
    elif "13M" in specstr_name:
        specstr = SpectrogramTransformer(output_dim=2, dropout=dropout, device=device).to(device)
    else:
        raise ValueError("Unknown specstr model size! There is two available models: 9M and 13M")
    specstr.load_state_dict(torch.load(os.path.join(specstr_path, specstr_name), weights_only=True))

    rawtr = None
    lirycstr = None
    transform_specs, min_amp, _ = get_specs_scaler()

    if rawtr_name is not None:
        rawtr_path = str(os.getenv("RAWTR_SAVE_PATH", "./outputs/"))
        rawtr = RawAudioTransformer(output_channels=2, dropout=dropout, device=device).to(device)
        rawtr.load_state_dict(torch.load(os.path.join(rawtr_path, rawtr_name), weights_only=True))
    
    if textr_name is not None:
        textr_path = str(os.getenv("TEXTR_SAVE_PATH", "./outputs/"))
        whisper_model_name = str(os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small"))
        text_extractor = TextExtractor(whisper_model_name=whisper_model_name, device=device).to(device)
        textr = TextSentimentTransformer(output_channels=2, whisper_model_name=whisper_model_name, device=device).to(device)
        textr.load_state_dict(torch.load(os.path.join(textr_path, textr_name), weights_only=True))
        lirycstr = LirycsSentimentTransformer(text_extractor, textr.transformer, device=device).to(device)

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

    learning_rate = float(os.getenv("FULL_LEARNING_RATE", 0.001))
    batch_size = int(os.getenv("FULL_BATCH_SIZE", 32))
    epochs = int(os.getenv("FULL_EPOCHS", 10))
    l2_reg = float(os.getenv("FULL_L2_REG", 0.01))
    max_text_len = int(os.getenv("FULL_MAX_TEXT_LEN", 128))
    kfold_splits = int(os.getenv("KFOLD_SPLITS", 5))
    dataset_name = f"dataset_{moods}_moods.tsv"
    # dataset_name = f"lyrics_supset_hs.tsv"

    audio_max_seq_len = int(os.getenv(f"RAWTR_MAX_SEQ_LEN", 50000))
    audio_min_seq_len = int(os.getenv(f"RAWTR_MIN_SEQ_LEN", 20000))
    sample_rate = int(os.getenv(f"RAWTR_SAMPLE_RATE", 22050))
    specs_max_seq_len = int(os.getenv(f"SPECTSR_MAX_SEQ_LEN", 5000))
    specs_min_seq_len = int(os.getenv(f"SPECTSR_MIN_SEQ_LEN", 2000))


    dataloader = KFoldHeterogeneousDataLoader(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        splits=kfold_splits,
        target_mode='onehot',
        use_augmentation=False,
        min_audio_seq_len=audio_min_seq_len,
        max_audio_seq_len=audio_max_seq_len,
        min_spec_seq_len=specs_max_seq_len,
        max_spec_seq_len=specs_min_seq_len,
        transform_spec=transform_specs,
        sample_rate=sample_rate,
        num_workers=4,
        moods=moods,
        random_state=random_state
    )

    rawtr_gruformer = None
    if rawtr is not None:
        rawtr_gruformer = rawtr.congruformer
    
    model = HeterogeneousDataSentimentClassifier(
        output_dim=2,
        specs_model=specstr.congruformer,
        text_model=lirycstr,
        audio_model=rawtr_gruformer,
        dropout=dropout,
        device=device
    ).to(device)

    print(f"Built model:\n", model)
    total_params, trainable_params = get_params_count(model)
    formated_total_params = format(total_params, ",").replace(",", " ")
    formated_trainable_params = format(trainable_params, ",").replace(",", " ")
    print(f"Total params: {formated_total_params}")
    print(f"Trainable params: {formated_trainable_params}\n")

    if task_type == "train" or task_type == "ctrain":
        trainer = ClassificationModelTrainer(
            model, model_name="full", num_classes=2, save_path=save_path,
            target_mode="onehot", kfold_loader=dataloader, lr=learning_rate,
            epochs=epochs, l2_reg=l2_reg
        )

        if task_type == "train":
            trainer.init_new_train()
        else:
            trainer.init_continue_train(model_name)
            
        trainer.train_model()
    else:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name), weights_only=True))
        test_loader, classes = dataloader.get_test_loader()
        evaluate_classification_model(model, classes, target_mode="onehot", test_loader=test_loader)
    