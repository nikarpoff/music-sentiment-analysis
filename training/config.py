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


# Available models.
PURE_SPECSTR = "pure_specstr"
SPECSTR = "specstr"
SPECS_AUTOENCODER = "specs_autoencoder"
PRETRAINED_SPECSTR = "pretrained_specstr"

RAWTR = "rawtr"
PRETRAINED_RAWTR = "pretrained_rawtr"

FEAMLP = "feamlp"

LIRYCSTR = "lirycstr"
TEXT_TR = "textr"

# Enumerations of models.
AVAILABLE_MODELS = [PURE_SPECSTR, SPECSTR, SPECS_AUTOENCODER, PRETRAINED_SPECSTR, RAWTR, PRETRAINED_RAWTR, FEAMLP, LIRYCSTR]
AUTOENCODER_MODELS = [SPECS_AUTOENCODER]
PRETRAINED_MODELS = [PRETRAINED_SPECSTR]
CLASSIFICATION_MODELS = [SPECSTR, PRETRAINED_SPECSTR, PURE_SPECSTR, RAWTR, PRETRAINED_RAWTR, FEAMLP, LIRYCSTR]
SPECS_MODELS = [SPECSTR, PRETRAINED_SPECSTR, PURE_SPECSTR, SPECS_AUTOENCODER]
RAW_AUDIO_MODELS = [RAWTR, PRETRAINED_RAWTR]
FEATURES_MODELS = [FEAMLP]
LIRYCS_MODELS = [LIRYCSTR]
TEXT_MODELS = [TEXT_TR]

# Avaliable target modes.
AUTOENCODER_TARGET = "autoenc"      # mode for x -> x regression
ONE_HOT_TARGET = "onehot"           # mode for x -> y single label classification
MULTILABEL_TARGET = "multilabel"    # mode for x -> y multilabel classification

REGRESSION_TARGETS = [AUTOENCODER_TARGET]
CLASSIFICATION_TARGETS = [ONE_HOT_TARGET, MULTILABEL_TARGET]

# Standards
TIMESTAMP_FORMAT = "%d%m%y_%H%M%S"
DATE_FORMAT = "%d_%m_%y"

