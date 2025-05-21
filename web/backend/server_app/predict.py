import os
import librosa
import numpy as np
import io
import soundfile as sf
from rest_framework import status
from backend.settings import BASE_DIR

MODELS_PATH = os.path.join(BASE_DIR, 'data')

# Use params like in MTG-Jamendo dataset.
FFT_WINDOW_LEN = 512
HOP_LEN = 256
WINDOW = 'hann'
N_MELS = 96


class Prediction:
    def predict(self, request):
        return_dict = {}
        result_dict = {}

        try:
            # In this request file required!
            if 'audio' not in request.FILES:
                raise ValueError("No audio part in the request.")

            audio_file = request.FILES['audio']

            # Read as binary buffer.
            audio_bytes = audio_file.read()
            audio_buffer = io.BytesIO(audio_bytes)

            # Upload file by soundfile. Use the same sample rate as in dataset.
            target_sr = 12000
            y, sr = librosa.load(audio_buffer, sr=target_sr)

            # Get spectrogram.
            mel_spec = librosa.feature.melspectrogram(y=y,
                                                      sr=sr,
                                                      n_fft=FFT_WINDOW_LEN,
                                                      hop_length=HOP_LEN,
                                                      n_mels=N_MELS,
                                                      window=WINDOW,
                                                      )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # mock prediction
            prediction = "sad"
            logits = [0.6, 0.4]

            result_dict['prediction'] = prediction
            result_dict['logits'] = logits
            result_dict['mel_shape'] = mel_spec_db.shape  # debug mode!

            return_dict['response'] = result_dict
            return_dict['status'] = status.HTTP_200_OK
            return return_dict

        except Exception as e:
            return_dict['response'] = "Exception when prediction: " + str(e)
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict
