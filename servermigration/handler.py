import runpod
import os
import torch
import tempfile
import base64
import numpy as np
import json
import logging
from io import BytesIO
from typing import Dict, Any, Union
import cv2
import mediapipe as mp
import time

# Import your model modules
from model import SLR
from VideoLoader import KeypointExtractor, read_video
from VideoDataset import process_keypoints
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Global variables for model and other resources
slr_model = None
keypoint_extractor = None
selected_keypoints = None
idx_to_word = None
device = None
gesture_recognizer = None

# Set up MediaPipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def initialize_models():
    """
    Initialize all models needed for sign language and gesture recognition
    Optimized for GPU utilization
    """
    global slr_model, keypoint_extractor, selected_keypoints, idx_to_word, device, gesture_recognizer
    
    start_time = time.time()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Initialize SLR model
    slr_model = SLR(
        n_embd=16*64, 
        n_cls_dict={'asl_citizen':2305, 'lsfb': 4657, 'wlasl':2000, 'autsl':226, 'rsl':1001},
        n_head=16, 
        n_layer=6,
        n_keypoints=63,
        dropout=0.6, 
        max_len=64,
        bias=True
    )
    
    # Compile model if using PyTorch 2.0+
    if hasattr(torch, 'compile'):
        try:
            slr_model = torch.compile(slr_model)
            logging.info("Model compiled successfully with torch.compile")
        except Exception as e:
            logging.warning(f"Model compilation failed, using non-compiled version: {e}")
    
    # Load model weights
    slr_model.load_state_dict(torch.load('./models/big_model.pth', map_location=device))
    slr_model.to(device)
    slr_model.eval()
    
    # Load keypoint extractor with GPU support
    use_gpu = device.type == 'cuda'
    keypoint_extractor = KeypointExtractor(use_gpu=use_gpu)
    
    # Define selected keypoints
    selected_keypoints = list(range(42)) 
    selected_keypoints = selected_keypoints + [x + 42 for x in ([291, 267, 37, 61, 84, 314, 310, 13, 80, 14] + [152])]
    selected_keypoints = selected_keypoints + [x + 520 for x in ([2, 5, 7, 8, 11, 12, 13, 14, 15, 16])]
    
    # Load gloss dictionary
    gloss_info = pd.read_csv('./gloss.csv')
    idx_to_word = {}
    for i in range(len(gloss_info)):
        idx_to_word[gloss_info['idx'][i]] = gloss_info['word'][i]
    
    # Load gesture recognizer with GPU delegate if available
    try:
        delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path='./gesture_recognizer.task',
                delegate=delegate  # Use GPU delegate when possible
            ),
            running_mode=VisionRunningMode.IMAGE,
        )
        gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)
        logging.info(f"Gesture recognizer loaded successfully with delegate: {delegate}")
    except Exception as e:
        logging.warning(f"Failed to load gesture recognizer: {e}")
        gesture_recognizer = None
    
    # GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logging.info(f"GPU Memory: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logging.info(f"GPU Memory: Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    end_time = time.time()
    logging.info(f"Model initialization completed in {end_time - start_time:.2f} seconds")
    
    return True

def recognize_sign_from_video(video_data: bytes) -> Dict[str, Any]:
    """
    Process video data from bytes and return the recognized sign language word
    Optimized for GPU performance
    """
    global slr_model, keypoint_extractor, selected_keypoints, idx_to_word, device
    
    # Ensure model is initialized
    if slr_model is None:
        initialize_models()
    
    try:
        start_time = time.time()
        logging.info(f"Processing video of size: {len(video_data) / 1024:.1f} KB")
        
        # Save incoming video data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            temp_path = tmp.name
        
        # Read the video file into a tensor
        video = read_video(temp_path)
        video = video.permute(0, 3, 1, 2)/255
        
        # Move video to the correct device
        video = video.to(device)
        
        # Extract keypoints using MediaPipe with GPU support
        pose = keypoint_extractor.extract_fast_parallel(video)
        height, width = video.shape[-2], video.shape[-1]
        
        # Reduce the number of samples for faster processing
        sample_amount = 8
        
        # Use CUDA Graphs for repeated computation patterns if available
        if device.type == 'cuda' and hasattr(torch.cuda, 'graphs') and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                
                # Warmup
                keypoints, valid_keypoints = process_keypoints(
                    pose, 64, selected_keypoints, height=height, width=width, 
                    augment=True, device=device
                )
                keypoints = keypoints.to(device)
                valid_keypoints = valid_keypoints.to(device)
                
                with torch.cuda.graph(g):
                    output = slr_model.heads['asl_citizen'](
                        slr_model(keypoints.unsqueeze(0), valid_keypoints.unsqueeze(0))
                    )
                
                # Execute actual computation
                logits = 0
                with torch.no_grad():
                    for i in range(sample_amount):
                        keypoints, valid_keypoints = process_keypoints(
                            pose, 64, selected_keypoints, height=height, width=width, 
                            augment=True, device=device
                        )
                        torch.cuda.synchronize()
                        g.replay()
                        logits = logits + output
                        
                logging.info("Used CUDA Graphs for inference")
                
            except Exception as e:
                logging.warning(f"CUDA Graphs failed: {e}, falling back to regular inference")
                # Fall back to regular inference
                logits = 0
                with torch.no_grad():
                    for i in range(sample_amount):
                        keypoints, valid_keypoints = process_keypoints(
                            pose, 64, selected_keypoints, height=height, width=width, 
                            augment=True, device=device
                        )
                        output = slr_model.heads['asl_citizen'](
                            slr_model(keypoints.unsqueeze(0), valid_keypoints.unsqueeze(0))
                        )
                        logits = logits + output
        else:
            # Regular inference without CUDA Graphs
            logits = 0
            with torch.no_grad():
                for i in range(sample_amount):
                    keypoints, valid_keypoints = process_keypoints(
                        pose, 64, selected_keypoints, height=height, width=width, 
                        augment=True, device=device
                    )
                    output = slr_model.heads['asl_citizen'](
                        slr_model(keypoints.unsqueeze(0), valid_keypoints.unsqueeze(0))
                    )
                    logits = logits + output
        
        # Get the top prediction
        idx = torch.argsort(logits, descending=True)[0].tolist()
        
        # Get top 3 predictions with scores
        top_3_indices = idx[:3] if isinstance(idx, list) else [idx]
        top_3_scores = logits[0][top_3_indices].tolist() if isinstance(idx, list) else [logits[0][idx].item()]
        
        # Format top 3 results
        results = [
            {"word": idx_to_word[index], "score": float(score)}
            for index, score in zip(top_3_indices, top_3_scores)
        ]
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Print performance information
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f"Inference completed in {inference_time:.2f} seconds")
        
        # GPU memory stats
        if torch.cuda.is_available():
            logging.info(f"GPU Memory: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        return {
            "recognized_word": idx_to_word[top_3_indices[0]],
            "top_results": results,
            "inference_time": inference_time
        }
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error(f"Error processing video: {str(e)}")
        raise Exception(f"Error processing video: {str(e)}")

def recognize_gesture(image_data: bytes) -> Dict[str, Any]:
    """
    Recognize gesture from image data
    Optimized for GPU performance
    """
    global gesture_recognizer, device
    
    # Ensure model is initialized
    if gesture_recognizer is None:
        initialize_models()
        if gesture_recognizer is None:
            return {"error": "Gesture recognizer failed to initialize"}
    
    try:
        start_time = time.time()
        logging.info(f"Processing image of size: {len(image_data) / 1024:.1f} KB")
        
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process with MediaPipe (GPU-accelerated via delegate)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = gesture_recognizer.recognize(mp_image)
        
        # Get detected gesture
        detected_gesture = "None"
        confidence = 0.0
        
        if results.gestures and len(results.gestures) > 0 and len(results.gestures[0]) > 0:
            detected_gesture = results.gestures[0][0].category_name
            confidence = results.gestures[0][0].score
        
        # Print performance information
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f"Gesture recognition completed in {inference_time:.2f} seconds")
        
        return {
            "gesture": detected_gesture,
            "confidence": float(confidence),
            "inference_time": inference_time
        }
    
    except Exception as e:
        logging.error(f"Error recognizing gesture: {str(e)}")
        raise Exception(f"Error recognizing gesture: {str(e)}")

def handler(event):
    """
    RunPod handler function - handles both sign recognition and gesture recognition
    """
    # Initialize models on first request
    if slr_model is None or gesture_recognizer is None:
        initialize_models()
    
    # Get the input data
    input_data = event.get("input", {})
    
    # Determine which endpoint to use based on input
    endpoint = input_data.get("endpoint", "recognize_sign_from_video")
    
    try:
        # Handle sign language recognition endpoint
        if endpoint == "recognize_sign_from_video":
            # Check if we have a video file in the input
            if "video_base64" in input_data:
                # Decode base64 video
                video_base64 = input_data.get("video_base64")
                video_data = base64.b64decode(video_base64)
                
                # Process video
                return recognize_sign_from_video(video_data)
            
            # Direct file upload handling (for testing)
            elif "video_url" in input_data:
                import requests
                video_url = input_data.get("video_url")
                try:
                    response = requests.get(video_url)
                    video_data = response.content
                    return recognize_sign_from_video(video_data)
                except Exception as e:
                    return {"error": f"Error downloading or processing video: {str(e)}"}
            
            else:
                return {"error": "No video provided. Please send base64 encoded video in the 'video_base64' field or a URL in 'video_url'."}
        
        # Handle gesture recognition endpoint
        elif endpoint == "recognize_gesture":
            if "image_base64" in input_data:
                # Decode base64 image
                image_base64 = input_data.get("image_base64")
                image_data = base64.b64decode(image_base64)
                
                # Process image
                return recognize_gesture(image_data)
            
            # Direct file upload handling (for testing)
            elif "image_url" in input_data:
                import requests
                image_url = input_data.get("image_url")
                try:
                    response = requests.get(image_url)
                    image_data = response.content
                    return recognize_gesture(image_data)
                except Exception as e:
                    return {"error": f"Error downloading or processing image: {str(e)}"}
            
            else:
                return {"error": "No image provided. Please send base64 encoded image in the 'image_base64' field or a URL in 'image_url'."}
        
        else:
            return {"error": f"Unknown endpoint: {endpoint}. Supported endpoints are 'recognize_sign_from_video' and 'recognize_gesture'."}
    
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}
    finally:
        # Clean up GPU memory when possible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Decide whether to preload models or load on first request
if os.environ.get("PRELOAD_MODELS", "false").lower() == "true":
    logging.info("Preloading models on startup...")
    initialize_models()
else:
    logging.info("Models will be loaded on first request")

# Start the serverless handler if run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})