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

# Enumerations of models.
AUTOENCODER_MODELS = [SPECS_AUTOENCODER]
PRETRAINED_MODELS = [PRETRAINED_SPECSTR]
CLASSIFICATION_MODELS = [SPECSTR, PRETRAINED_SPECSTR, PURE_SPECSTR]
SPECS_MODELS = [SPECSTR, PRETRAINED_SPECSTR, PURE_SPECSTR, SPECS_AUTOENCODER]

# Avaliable target modes.
AUTOENCODER_TARGET = "autoenc"      # mode for x -> x regression
ONE_HOT_TARGET = "onehot"           # mode for x -> y single label classification
MULTILABEL_TARGET = "multilabel"    # mode for x -> y multilabel classification

REGRESSION_TARGETS = [AUTOENCODER_TARGET]
CLASSIFICATION_TARGETS = [ONE_HOT_TARGET, MULTILABEL_TARGET]

# Standards
TIMESTAMP_FORMAT = "%d%m%y_%H%M%S"
DATE_FORMAT = "%d_%m_%y"

