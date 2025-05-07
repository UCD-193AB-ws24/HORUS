from datetime import datetime
from functools import lru_cache
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # need this to suppress the mp errors

import mediapipe as mp
import time
# import pyaudio
import wave
import requests
import aiohttp
import pandas as pd
import torch
import uvicorn
import whisper
from VideoLoader import KeypointExtractor, read_video
from VideoDataset import process_keypoints
from model import SLR
from pydantic import BaseModel
import shutil
import datetime

import torch._dynamo
torch._dynamo.config.suppress_errors = True

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

class Item (BaseModel):
    signed: str
    translated: str
    video: bool

recognizer = GestureRecognizer.create_from_options(options)

whisper_model = whisper.load_model("tiny")


model = SLR(
    n_embd=16*64, 
    n_cls_dict={'asl_citizen':2305, 'lsfb': 4657, 'wlasl':2000, 'autsl':226, 'rsl':1001},
    n_head=16, 
    n_layer=6,
    n_keypoints=63,
    dropout=0.6, 
    max_len=64,
    bias=True
)

model = torch.compile(model)
model.load_state_dict(torch.load('./models/big_model.pth', map_location=torch.device('cpu')))
model.eval()

gloss_info = pd.read_csv('./gloss.csv')
idx_to_word = {}
for i in range(len(gloss_info)):
    idx_to_word[gloss_info['idx'][i]] = gloss_info['word'][i]

@lru_cache(maxsize=1)
def get_selected_keypoints():
    selected_keypoints = list(range(42)) 
    selected_keypoints = selected_keypoints + [x + 42 for x in ([291, 267, 37, 61, 84, 314, 310, 13, 80, 14] + [152])]
    selected_keypoints = selected_keypoints + [x + 520 for x in ([2, 5, 7, 8, 11, 12, 13, 14, 15, 16])]
    return selected_keypoints

@lru_cache(maxsize=1)
def get_keypoint_extractor():
    return KeypointExtractor()


@app.post("/recognize-sign-from-video/")
async def recognize_sign_from_video(file: UploadFile = File(...)):
    """
    Receives an MP4 video file and returns the recognized sign language word
    """
    if not file:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Generate a unique filename with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"uploaded_video_{timestamp}.mp4"
    
    # # Save the file in the current directory
    # file_path = os.path.join(os.getcwd(), filename)
    
    # # Write the uploaded file to disk
    # contents = await file.read()
    # with open(file_path, "wb") as f:
    #     f.write(contents)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name
    
    try:
        # Read the video file into a tensor
        video = read_video(temp_path)
        video = video.permute(0, 3, 1, 2)/255
        video = torch.flip(video, dims=[-2])
        # Extract keypoints using MediaPipe
        keypoint_extractor = get_keypoint_extractor()
        
        # Use the faster extract_fast method
        pose = keypoint_extractor.extract_fast_parallel(video)
        height, width = video.shape[-2], video.shape[-1]
        
        # Define the selected keypoints (same as in run_model.ipynb)
        selected_keypoints = get_selected_keypoints()  
        # Reduce the number of samples for faster processing
        sample_amount = 8  # Reduced from 16 for speed
        
        logits = 0
        with torch.no_grad():
            model.eval()
            for i in range(sample_amount):
                keypoints, valid_keypoints = process_keypoints(pose, 64, selected_keypoints, height=height, width=width, augment=True)
                logits = logits + model.heads['asl_citizen'](model(keypoints.unsqueeze(0), valid_keypoints.unsqueeze(0)))
        
        # Get the top prediction
        idx = torch.argsort(logits, descending=True)[0].tolist()
        
        if isinstance(idx, int):
            # If it's already an integer, use it directly
            top_word = idx_to_word[idx]
        else:
            # If it's a list, take the first element
            top_word = idx_to_word[idx[0]]
        
        return {"recognized_word": top_word}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/test-sign-recognition/{video_filename}")
async def test_sign_recognition(video_filename: str, request: Request):
    """
    Test route that reads an MP4 file from the 'test_videos' directory 
    and sends it to the recognize-sign-from-video endpoint
    """
    video_path = os.path.join("test_videos", video_filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file {video_filename} not found in test_videos directory")
    
    # Create a client and make request to our own endpoint
    host = request.headers.get('host')
    url = f"http://{host}/recognize-sign-from-video/"
    
    # Use aiohttp for async requests instead of synchronous requests
    async with aiohttp.ClientSession() as session:
        with open(video_path, "rb") as video_file:
            data = aiohttp.FormData()
            data.add_field('file', 
                           video_file, 
                           filename=video_filename,
                           content_type='video/mp4')
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status, 
                        detail=f"Error from recognition endpoint: {error_text}"
                    )

@app.post("/recognize-gesture/")
async def recognize_gesture(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        # contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        results = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
        detected_gesture = results.gestures[0][0].category_name if results.gestures else "None"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        result = whisper_model.transcribe(temp_path, language='en', fp16 = False)  # or omit language if auto-detect
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

@app.post("/send_help_form/")
async def send_help_form(item: Item):
    if not item:
        raise HTTPException(status_code=400, detail=f"No help form provided")
    
    if item.video:
        video_file = "sign_language.mp4"
        if os.path.isfile(video_file):
            ct = datetime.datetime.now()
            new_video_file_name = item.signed + str(ct) + ".mp4"
            os.rename(video_file, new_video_file_name)
            destination = "./error_videos/"
            shutil.move("./" + new_video_file_name, destination)

    return {"message": "File received"}

@app.get("/")
def read_root():
    return {"message": "Server is running!"}


# def record_audio_and_send(
#     server_url="http://127.0.0.1:8000/process_audio/",
#     duration=5,
#     output_filename="output.wav"
# ):
#     """
#     Records audio from your microphone for `duration` seconds,
#     saves to `output_filename`, and sends it to the `server_url`
#     endpoint for STT.
#     """

#     # Audio recording parameters
#     chunk = 1024          # Number of frames per buffer
#     format = pyaudio.paInt16
#     channels = 1
#     rate = 16000          # Whisper often expects 16k, but you can do 44100
#     record_seconds = duration

#     # Initialize PyAudio
#     p = pyaudio.PyAudio()

#     # Open stream
#     stream = p.open(format=format,
#                     channels=channels,
#                     rate=rate,
#                     input=True,
#                     frames_per_buffer=chunk)

#     print(f"Recording for {record_seconds} second(s)...")
#     frames = []

#     # Read mic data
#     for _ in range(0, int(rate / chunk * record_seconds)):
#         data = stream.read(chunk)
#         frames.append(data)

#     # Stop and close
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     print("Recording complete. Saving WAV file...")

#     # Save as WAV
#     wf = wave.open(output_filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(format))
#     wf.setframerate(rate)
#     wf.writeframes(b''.join(frames))
#     wf.close()

#     # Send the file to the server
#     try:
#         with open(output_filename, "rb") as f:
#             files = {"file": ("recording.wav", f, "audio/wav")}
#             print(f"Sending to {server_url} ...")
#             response = requests.post(server_url, files=files)
#         if response.status_code == 200:
#             data = response.json()
#             print("Server response:", data)
#         else:
#             print(f"Error from server: HTTP {response.status_code}")
#             print("Response:", response.text)
#     finally:
#         # Clean up local WAV file if you want
#         if os.path.exists(output_filename):
#             os.remove(output_filename)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace with your actual module:app
        host="127.0.0.1",  # Listen on all interfaces
        port=8001,
    )

        #     ssl_keyfile="./certs/key.pem",
        #     ssl_certfile="./certs/cert.pem"