import numpy as np
import cv2
from PIL import Image, ExifTags
import time

from fastapi import File, FastAPI, UploadFile, HTTPException
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import MaskRCNN

app = FastAPI(
    title="Angle detection",
    version="0.1.0",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

predictor = MaskRCNN()

@app.post("/predict", tags=['Predict'])
def predict(file: UploadFile = File(...)):
    t = time.time()
    try:
        image = Image.open(file.file)

    except:
        raise HTTPException(status_code=400, detail="An image file is not appropriate!")

    img = np.array(image)
    result = predictor.predict(img)

    print(time.time() - t)
    return { 
        "result": result
        }
    

@app.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url="/docs/")
