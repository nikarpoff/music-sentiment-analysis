import librosa
import numpy as np
import io


# Use params like in MTG-Jamendo dataset.
FFT_WINDOW_LEN = 512
HOP_LEN = 256
WINDOW = 'hann'
N_MELS = 96

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

    return mel_spec_db
