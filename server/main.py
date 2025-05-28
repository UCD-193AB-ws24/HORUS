# Set environment variables BEFORE importing anything else
import os
import sys

# Critical: Set these before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_v'] = '-1'
os.environ['GLOG_minloglevel'] = '3'

import torch
# Set optimal OpenMP configuration
def setup_openmp_for_interpolation():
    """Configure OpenMP settings for optimal interpolation performance"""
    cpu_count = os.cpu_count() or 4
    thread_count = min(cpu_count - 1, 6)  # Conservative for stability
    
    os.environ['OMP_SCHEDULE'] = 'static'
    os.environ['OMP_PROC_BIND'] = 'close'
    os.environ['OMP_PLACES'] = 'cores'
    
    torch.set_num_threads(thread_count)
    print(f"OpenMP configured for {thread_count} threads")

# Call setup function
setup_openmp_for_interpolation()

# Optimized threading - safe for keypoint extraction
os.environ['OMP_NUM_THREADS'] = '4'  # Reasonable threading
os.environ['MKL_NUM_THREADS'] = '4'  # Reasonable threading  
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Reasonable threading
os.environ['NUMEXPR_NUM_THREADS'] = '4'  # Reasonable threading

# Memory allocation settings
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'

# Disable threading in various libraries that might conflict
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datetime import datetime
from functools import lru_cache
import tempfile
import threading
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np 
import mediapipe as mp
import aiohttp
import pandas as pd
#import torch
import whisper
from VideoLoader import KeypointExtractor, read_video
from VideoDataset import process_keypoints
from model import SLR
from pydantic import BaseModel
import shutil
import uvicorn
import datetime
import gc
import warnings
warnings.filterwarnings("ignore")

# Torch settings for memory safety
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_num_threads(4)  # Reasonable PyTorch threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lock for MediaPipe operations to prevent race conditions
mediapipe_lock = threading.Lock()

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Memory-safe MediaPipe options
options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path='./models/gesture_recognizer.task',
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.IMAGE,
)

class Item(BaseModel):
    signed: str
    translated: str
    video: bool

# Initialize models with lock protection
with mediapipe_lock:
    recognizer = GestureRecognizer.create_from_options(options)

# Load other models
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

# Load the compiled model weights and fix the key names
def load_compiled_model_weights(model, checkpoint_path):
    """Load weights from a torch.compile() saved model"""
    try:
        # Load the state dict
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Check if this is a compiled model (has _orig_mod prefix)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected compiled model, fixing key names...")
            # Remove the _orig_mod. prefix from all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load the cleaned state dict
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try alternative loading methods
        try:
            # Method 2: Load with strict=False
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with strict=False")
            return model
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise e

# Load the model with the fixed function
model = load_compiled_model_weights(model, './models/big_model.pth')
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

# Thread-safe keypoint extractor
keypoint_extractor = None
extractor_lock = threading.Lock()

def get_keypoint_extractor():
    global keypoint_extractor
    with extractor_lock:
        if keypoint_extractor is None:
            keypoint_extractor = KeypointExtractor()
        return keypoint_extractor

@app.post("/recognize-sign-from-video/")
async def recognize_sign_from_video(file: UploadFile = File(...)):
    """
    Memory-safe video processing with proper resource management
    """
    if not file:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    temp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name
        
        # Process video with memory safety
        result = await process_video_safe(temp_path)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        # Force cleanup
        gc.collect()

async def process_video_safe(video_path: str):
    """
    Process video with comprehensive memory management
    """
    video = None
    pose = None
    
    try:
        # Read video
        video = read_video(video_path)
        if video is None:
            raise ValueError("Could not read video file")
            
        # Preprocess video
        video = video.permute(0, 3, 1, 2) / 255.0
        video = torch.flip(video, dims=[-2])
        
        print(f"Processing video with shape: {video.shape}")
        
        # Get extractor (thread-safe)
        extractor = get_keypoint_extractor()
        
        # Extract keypoints with single-threaded processing to avoid memory corruption
        try:
            # Use the safe sequential extractor (no parallel processing)
            pose = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: extractor.extract_safe_parallel(video)
            )
        except Exception as e:
            print(f"Keypoint extraction failed: {e}")
            raise ValueError("Failed to extract keypoints from video")
        
        if pose is None or len(pose) == 0:
            raise ValueError("No keypoints extracted from video")
            
        height, width = video.shape[-2], video.shape[-1]
        
        print("Video shape:", video.shape)
        print("Pose shape:", pose.shape)
        print("Pose sample (frame 0):", pose[0][:5] if len(pose) > 0 else "Empty")
        
        # Process keypoints for model
        selected_keypoints = get_selected_keypoints()
        sample_amount = 16  # Reduced further for memory safety
        
        logits = None
        try:
            with torch.no_grad():
                model.eval()
                
                for i in range(sample_amount):
                    keypoints, valid_keypoints = process_keypoints(
                        pose, 64, selected_keypoints, 
                        height=height, width=width, augment=True
                    )
                    
                    if keypoints.numel() == 0:
                        continue
                    
                    # Single batch inference
                    batch_logits = model.heads['asl_citizen'](
                        model(keypoints.unsqueeze(0), valid_keypoints.unsqueeze(0))
                    )
                    
                    if logits is None:
                        logits = batch_logits
                    else:
                        logits = logits + batch_logits
                    
                    # Clear intermediate tensors
                    del keypoints, valid_keypoints, batch_logits
                    
        except Exception as e:
            print(f"Model inference error: {e}")
            raise ValueError(f"Model inference failed: {str(e)}")
        
        if logits is None:
            raise ValueError("No valid keypoints for model inference")
        
        # Get prediction
        try:
            top_idx = torch.argmax(logits).item()  # Get index of maximum value
            top_word = idx_to_word.get(top_idx, "UNKNOWN")
        except Exception as e:
            print(f"Prediction error: {e}")
            top_word = "UNKNOWN"
        
        return {"recognized_word": top_word}
        
    finally:
        # Explicit cleanup
        del video, pose
        gc.collect()

@app.get("/test-sign-recognition/{video_filename}")
async def test_sign_recognition(video_filename: str, request: Request):
    """
    Test route with proper error handling
    """
    video_path = os.path.join("test_videos", video_filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file {video_filename} not found")
    
    host = request.headers.get('host')
    url = f"http://{host}/recognize-sign-from-video/"
    
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
                        detail=f"Error: {error_text}"
                    )

@app.post("/recognize-gesture/")
async def recognize_gesture(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Thread-safe MediaPipe operation
        with mediapipe_lock:
            results = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
            
        detected_gesture = results.gestures[0][0].category_name if results.gestures else "None"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    return {"gesture": detected_gesture}

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Audio processing with memory safety
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        result = whisper_model.transcribe(temp_path, language='en', fp16=False)
        recognized_text = result.get("text", "")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    return {"recognized_text": recognized_text}

@app.post("/send_help_form/")
async def send_help_form(item: Item):
    if not item:
        raise HTTPException(status_code=400, detail="No help form provided")
    
    if item.video:
        video_file = "sign_language.mp4"
        if os.path.isfile(video_file):
            ct = datetime.datetime.now()
            new_video_file_name = item.signed + str(ct) + ".mp4"
            os.rename(video_file, new_video_file_name)
            destination = "./error_videos/"
            shutil.move("./" + new_video_file_name, destination)

    return {"message": "File received"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    import psutil
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
    }

@app.get("/")
def read_root():
    return {"message": "Server is running safely!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=port,
        workers=1,  # Single worker to prevent memory conflicts
    )