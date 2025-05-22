import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import utils
from datetime import datetime
from modelarch import *
from fastapi import FastAPI, File, UploadFile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_NAME = "specstr9M-10.pth"
CLASSES_JSON_NAME = f"classes_onehot_specstr9M-10.json"
OUTPUT_DIMS = 4

TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"

app = FastAPI()

# Model loading.
print('Loading model...')
model = SpectrogramTransformer9M(OUTPUT_DIMS, device='cuda')

model.load_state_dict(torch.load(os.path.join(DATA_PATH, MODEL_NAME), weights_only=True))
model.eval()
print('Model loaded.')

classes = json.load(open(os.path.join(DATA_PATH, CLASSES_JSON_NAME), 'r'))

@app.post("/api/model/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    spec = utils.get_mel_spec(audio_bytes)
    spec = torch.from_numpy(spec).float()

    start_time = datetime.now()
    print(f"{start_time.strftime(TIMESTAMP_FORMAT)}: POST: spec shape: {spec.shape}")

    spec.to(model.device, non_blocking=True)
    spec = spec.unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(spec), dim=1).cpu().numpy().flatten()
    
    end_time = datetime.now()
    print(f"{end_time.strftime(TIMESTAMP_FORMAT)}: POST: prediction time: {end_time - start_time}")

    prediction = classes[probs.argmax()]
    
    return {"predict": prediction, "logits": probs.tolist(), "classes": classes}
