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
async def process_audio(audio: UploadFile = File(...)):
    """
    Receives an audio file and returns the transcribed text using Whisper (local).
    """
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Read file contents
        contents = await audio.read()
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

# @app.post("/process_audio/")
# async def process_audio(file: UploadFile = File(...)):
#     # Check that we have a file
#     contents = await file.read()
#     return {"msg": "File received", "file_size": len(contents)}

@app.get("/")
def read_root():
    return {"message": "Server is running!"}


import pyaudio
import wave
import requests
import time
import os

def record_audio_and_send(
    server_url="http://127.0.0.1:8000/process_audio/",
    duration=5,
    output_filename="output.wav"
):
    """
    Records audio from your microphone for `duration` seconds,
    saves to `output_filename`, and sends it to the `server_url`
    endpoint for STT.
    """

    # Audio recording parameters
    chunk = 1024          # Number of frames per buffer
    format = pyaudio.paInt16
    channels = 1
    rate = 16000          # Whisper often expects 16k, but you can do 44100
    record_seconds = duration

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"Recording for {record_seconds} second(s)...")
    frames = []

    # Read mic data
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording complete. Saving WAV file...")

    # Save as WAV
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Send the file to the server
    try:
        with open(output_filename, "rb") as f:
            files = {"file": ("recording.wav", f, "audio/wav")}
            print(f"Sending to {server_url} ...")
            response = requests.post(server_url, files=files)
        if response.status_code == 200:
            data = response.json()
            print("Server response:", data)
        else:
            print(f"Error from server: HTTP {response.status_code}")
            print("Response:", response.text)
    finally:
        # Clean up local WAV file if you want
        if os.path.exists(output_filename):
            os.remove(output_filename)


if __name__ == "__main__":
    record_audio_and_send(
        server_url="http://127.0.0.1:8000/process_audio/",  # or your LAN IP
        duration=5,
        output_filename="temp_audio.wav"
    )