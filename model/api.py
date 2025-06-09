import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
from datetime import datetime
from modelarch import *
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import MinMaxScaler

import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')

MODEL_NAME_HS = "specstr_hs_13M.pth"
MODEL_NAME_RE = "specstr_re_8M.pth"

CLASSES_JSON_NAME_HS = f"classes_hs.json"
CLASSES_JSON_NAME_RE = f"classes_re.json"
OUTPUT_DIMS = 2

TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"

app = FastAPI()

# Model loading.
print('Loading model...')
model_hs = SpectrogramTransformer(OUTPUT_DIMS, dropout=0., device='cuda').to('cuda')
model_re = SmallSpectrogramTransformer(OUTPUT_DIMS, dropout=0., device='cuda').to('cuda')

model_hs.load_state_dict(torch.load(os.path.join(DATA_PATH, MODEL_NAME_HS), weights_only=True))
model_re.load_state_dict(torch.load(os.path.join(DATA_PATH, MODEL_NAME_RE), weights_only=True))
model_hs.eval()
model_re.eval()

print('Model loaded.')

classes_hs = json.load(open(os.path.join(DATA_PATH, CLASSES_JSON_NAME_HS), 'r'))
classes_re = json.load(open(os.path.join(DATA_PATH, CLASSES_JSON_NAME_RE), 'r'))
classes = classes_hs + classes_re

scaler = MinMaxScaler(feature_range=(0, 1))


@app.post("/api/model/predict")
async def predict(file: UploadFile = File(...)):
    # Load audio in bytes.
    audio_bytes = await file.read()

    # Form mel-spectrogram
    spec = utils.get_mel_spec(audio_bytes)
    spec = scaler.fit_transform(spec)
    spec = torch.from_numpy(spec).float().unsqueeze(0)

    # Log current time.
    start_time = datetime.now()
    print(f"{start_time.strftime(TIMESTAMP_FORMAT)}: POST: spec shape: {spec.shape}")

    # Make prediction.
    hs_probs = utils.inference(model_hs, spec.to(model_hs.device, non_blocking=True))
    re_probs = utils.inference(model_re, spec.to(model_re.device, non_blocking=True))
    
    # Log prediction time.
    end_time = datetime.now()
    print(f"{end_time.strftime(TIMESTAMP_FORMAT)}: POST: prediction time: {end_time - start_time}")

    # Form and send response.
    probabilities = np.concatenate((hs_probs, re_probs), axis=0)
    print(probabilities)
    prediction = classes[probabilities.argmax()]
    
    return {"predict": prediction, "probs": probabilities.tolist(), "classes": classes}
