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

import torch
from torch import nn

class HeterogeneousDataSentimentClassifier(nn.Module):
    """
    Heterogeneous model that combines different types of models and input data.

    Base of the model - mel-spectrograms model. It is required part of the full model.
    Additional models can be added for text, audio, and numeric features.
    """
    def __init__(self,
                 output_dim,
                 specs_model: nn.Module,
                 text_model: nn.Module | None = None,
                 audio_model: nn.Module | None = None,
                 dropout: float = 0.2,
                 device='cuda'):
        super().__init__()
        self.specs_model = specs_model
        
        for param in self.specs_model.parameters():
            param.requires_grad = False

        self.text_model = text_model
        self.audio_model = audio_model
        
        self.depth = specs_model.depth if hasattr(specs_model, 'depth') else 256

        if text_model is not None:
            self.depth += text_model.depth if hasattr(text_model, 'depth') else 256
            
        if audio_model is not None:
            for param in self.audio_model.parameters():
                param.requires_grad = False
            self.depth += audio_model.depth if hasattr(audio_model, 'depth') else 256

        self.output_projection = nn.Sequential(
            nn.Linear(self.depth, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, output_dim)
        )

        self.device = device

    def forward(self, audio_input, spec_input) -> torch.Tensor:
        spec_features = self.specs_model(spec_input)
        combined_features = spec_features

        if self.text_model is not None:
            text_features = self.text_model(audio_input)            # text input extracted from raw audio (e.g., Whisper)
            combined_features = torch.cat((combined_features, text_features), dim=1)

        if self.audio_model is not None:
            audio_features = self.audio_model(audio_input)
            combined_features = torch.cat((combined_features, audio_features), dim=1)

        logits = self.output_projection(combined_features)
        return logits
