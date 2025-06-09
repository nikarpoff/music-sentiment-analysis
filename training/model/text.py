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


from torch import nn
from model.layer import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer


class TextTransformer(nn.Module):
    """
    Transformer with positional sinusoidal encoding and attention pooling.
    """
    def __init__(self, depth: int, nheads: int, num_encoders: int, dropout=0.2, whisper_model_name: str = "openai/whisper-small", device='cuda'):
        super().__init__()
        self.depth = depth
        self.device = device

        self.tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)
        vocab_size = self.tokenizer.vocab_size + len(self.tokenizer.added_tokens_encoder)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=depth)

        for param in self.embedding.parameters():
            param.requires_grad = False

        self.pos_encoding = PositionalEncoding(d_model=depth)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=depth,
            dim_feedforward=depth * 4,
            nhead=nheads,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True     # LayerNorm first
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.attention_pool = MultiHeadPool(depth, nheads, dropout)

    def forward(self, tokens_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens_ids: indices of tokens in the text sequence (batch_size, seq_len)
        padding_mask: mask for the text sequence (batch_size, seq_len): 1 for true tokens, 0 for paddings/special tokens
        """
        x = self.embedding(tokens_ids)  # (batch_size, seq_len, depth)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~padding_mask.bool())
        x = self.attention_pool(x)
        return x
    

class TextExtractor(nn.Module):
    def __init__(self, max_seq_len=128, whisper_model_name: str = "openai/whisper-small", device='cuda'):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len

        # Model params
        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: raw mono audio data (batch_size, seq_len) with values between ([-1.0, +1.0]), 16 kHz.
        Returns:
          - generated_ids: (batch_size, text_len) — generated text token indexes.
          - padding_mask: torch.LongTensor (batch_size, text_len) — mask with zero in padding/special tokens.
        """
        x = x.squeeze(1).to("cpu")

        list_of_x_els = [x_i for x_i in x.cpu().numpy()]

        p = self.processor(list_of_x_els, sampling_rate=16000, return_tensors="pt")
        input_features = p.input_features.to(self.device)
        
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=128,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            use_cache=True,
        )

        pad_id = self.processor.tokenizer.pad_token_id
        padding_mask = (generated_ids != pad_id).long()

        return generated_ids, padding_mask
    

class TextSentimentTransformer(nn.Module):
    """
    Text sentiment analisys transformer model.
    """
    def __init__(self, output_channels: int, dropout=0.2, whisper_model_name: str = "openai/whisper-small", device='cuda'):
        super().__init__()
        self.device = device

        # Model params
        TRANSFORMER_DEPTH = 312
        NHEAD = 6
        NUM_ENCODERS = 4

        self.transformer = TextTransformer(TRANSFORMER_DEPTH, NHEAD, NUM_ENCODERS, dropout, whisper_model_name, device=device)
        self.output_proj = nn.Sequential(
            nn.Linear(TRANSFORMER_DEPTH, TRANSFORMER_DEPTH // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(TRANSFORMER_DEPTH // 2, output_channels)
        )

    def forward(self, tokens_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens_ids: indices of tokens in the text sequence (batch_size, seq_len)
        padding_mask: mask for the text sequence (batch_size, seq_len): 1 for true tokens, 0 for paddings/special tokens
        """
        x = self.transformer(tokens_ids, padding_mask)
        return self.output_proj(x)


class LirycsSentimentTransformer(nn.Module):
    """
    Predict lirycs sentiment using pipeline: TextExtractor -> TransformerWithPooling -> Output Projection.
    Input data is raw audio, output is sentiment logits (output_channels,).
    """
    def __init__(self, text_extractor: TextExtractor, transformer: TextTransformer, device='cuda'):
        super().__init__()
        self.device = device
        self.text_extractor = text_extractor
        self.transformer = transformer
        self.depth = transformer.depth

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        text_ids, padding_mask = self.text_extractor(waveform)
        return self.transformer(text_ids, padding_mask)
