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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import uuid
import torch
import numpy as np
from datetime import datetime
from modelarch import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')

MODEL_NAME_HS = "specstr_hs_13M.pth"
MODEL_NAME_RE = "specstr_re_TEST.pth"

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

specs_processor = utils.SpectrogramProcessor(device="cuda")

@app.post("/api/model/predict")
async def predict(file: UploadFile = File(...)):
    request_uuid = uuid.uuid4()

    # Load audio in bytes.
    audio_bytes = await file.read()

    # Form mel-spectrogram
    try:
        spec = specs_processor.pipeline(audio_bytes, request_uuid)
    except ValueError as e:
        print(f"{datetime.now().strftime(TIMESTAMP_FORMAT)}: POST: Error processing audio at request {request_uuid}: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # Log current time.
    start_time = datetime.now()
    print(f"{start_time.strftime(TIMESTAMP_FORMAT)}: POST: spec shape: {spec.shape}; uuid: {request_uuid}")

    # Make prediction.
    hs_probs = utils.inference(model_hs, spec.to(model_hs.device, non_blocking=True))
    re_probs = utils.inference(model_re, spec.to(model_re.device, non_blocking=True))
    
    # Log prediction time.
    end_time = datetime.now()
    print(f"{end_time.strftime(TIMESTAMP_FORMAT)}: POST: prediction time: {end_time - start_time}; uuid: {request_uuid}")

    # Form and send response.
    probabilities = np.concatenate((hs_probs, re_probs), axis=0)
    prediction = classes[probabilities.argmax()]
    
    return {"predict": prediction, "probs": probabilities.tolist(), "classes": classes}
