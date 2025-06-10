import torch
from torch import nn
import torch.functional as F
import torchaudio
import numpy as np
import io


class SpectrogramProcessor:
    """
    Functionality to load and preprocess mel-spectrograms due torch.
    Use the same params for mel-spec generation as in target MTG-Jamendo dataset.
    """
    def __init__(self,
                 target_sample_rate=12000,
                 fft_window_len=512,
                 hop_len=256,
                 n_mels=96,
                 min_db_value=-90.0,
                 max_db_value=40.,
                 inference_window_len: int = 4096,
                 inference_hop_len: int = 2048,
                 inference_max_hops: int = 8,
                 device="cuda"):
        self.target_sample_rate = target_sample_rate
        self.min_db_value = min_db_value
        self.max_db_value = max_db_value
        self.window_len = inference_window_len
        self.hop_len = inference_hop_len
        self.max_hops = inference_max_hops
        self.device = device

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=fft_window_len,
            hop_length=hop_len,
            n_mels=n_mels,
            window_fn=torch.hann_window,
        ).to(device)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)

    def pipeline(self, audio_bytes):
        spec = self.get_mel_spec(audio_bytes)
        spec = self.minmax_scale_tensor(spec)
        return self.preprocess_spectrogram(spec)

    def get_mel_spec(self, audio_bytes):
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate).to(self.device)
        
        spec = self.mel_transform(waveform)
        spec_db = self.db_transform(spec)

        return spec_db

    def minmax_scale_tensor(self, spec_db: torch.Tensor) -> torch.Tensor:
        spec_db = spec_db.clamp(min=self.min_db_value, max=self.max_db_value)
        return (spec_db - self.min_db_value) / (self.max_db_value - self.min_db_value)

    def preprocess_spectrogram(self, spec: np.array,) -> torch.Tensor:
        """
        Batch the mel-spectrogram throught windows.
        """
        mels_n, length = spec.size(1), spec.size(-1)

        print(torch.min(spec), torch.max(spec), "\n")

        # Pad if needed
        if length < self.window_len:
            spec = F.pad(spec, (0, self.window_len - length))
            length = self.window_len

        # Compute number of windows
        num = min(self.max_hops, 1 + (length - self.window_len) // self.hop_len)

        # Create strided tensor of shape (num, mels_n, window_len)
        strides = spec.stride()
        windowed = spec.as_strided(
            size=(num, mels_n, self.window_len),
            stride=(self.hop_len * strides[-1], strides[-2], strides[-1])
        )

        return windowed

def inference(model: nn.Module, spec: torch.Tensor) -> np.array:
    with torch.no_grad():
        probabilities = torch.softmax(model(spec), dim=1).cpu().numpy()

    return np.mean(probabilities, axis=0)
