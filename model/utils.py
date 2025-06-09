import torch
from torch import nn
import librosa
import numpy as np
import io


# Use params like in MTG-Jamendo dataset.
FFT_WINDOW_LEN = 512
HOP_LEN = 256
WINDOW = 'hann'
N_MELS = 96

# Length of spectrogram to model observe
SPEC_WINDOW_LEN = 4096
SPEC_HOP_LEN = 2048
MAX_HOPS = 8

def get_mel_spec(audio_bytes):
    # Read as binary buffer.
    audio_buffer = io.BytesIO(audio_bytes)

    # Upload file by librosa. Use the same sample rate as in dataset.
    target_sr = 12000
    y, sr = librosa.load(audio_buffer, sr=target_sr)
    # Get spectrogram.
    mel_spec = librosa.feature.melspectrogram(y=y,
                                              sr=sr,
                                              n_fft=FFT_WINDOW_LEN,
                                              hop_length=HOP_LEN,
                                              n_mels=N_MELS,
                                              window=WINDOW,)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    print(np.min(mel_spec_db))
    print(np.max(mel_spec_db))

    return mel_spec_db

def inference(model: nn.Module, spec: torch.Tensor) -> np.array:
    current_step = 0
    current_timestamp = 0
    total_timestamps = spec.shape[-1]

    probabilities = []

    with torch.no_grad():
        while current_timestamp < total_timestamps and current_step < MAX_HOPS:
            windowed_spec = spec[:, :, current_timestamp:current_timestamp + SPEC_WINDOW_LEN]
            probabilities.append(torch.softmax(model(windowed_spec), dim=1).cpu().numpy().flatten())
            current_timestamp += SPEC_HOP_LEN
            current_step += 1
    
    if len(probabilities) == 0:
        print("Audio too short to predict!")
        return np.array[0., 0.]

    return np.mean(probabilities, axis=0)
