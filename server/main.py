import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import time
import whisper

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
)

recognizer = GestureRecognizer.create_from_options(options)

whisper_model = whisper.load_model("tiny")

@app.post("/recognize-gesture/")
async def recognize_gesture(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    results = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    
    detected_gesture = results.gestures[0][0].category_name if results.gestures else "None"
    print(detected_gesture)
    return {"gesture": detected_gesture}

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Receives an audio file and returns the transcribed text using Whisper (local).
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Read file contents
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        result = whisper_model.transcribe(temp_path, language='en')  # or omit language if auto-detect
        recognized_text = result.get("text", "")
    except Exception as e:
        # Cleanup and re-raise as HTTPException
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"recognized_text": recognized_text}

@app.get("/")
def read_root():
    return {"message": "Server is running!"}