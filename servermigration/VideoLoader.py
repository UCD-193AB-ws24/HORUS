import logging
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp_threading
from mediapipe.framework.formats import landmark_pb2

# Configure mediapipe components
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables for landmark target indices
hand_target_landmarks = list(range(21))
face_target_landmarks = list(range(478))
pose_target_landmarks = list(range(33))

# Function to load model buffers with error handling
def load_model_buffer(model_path):
    try:
        with open(model_path, "rb") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return None

# Lazy loading of model buffers
_gesture_model_buffer = None
_face_model_buffer = None

def get_gesture_model_buffer():
    global _gesture_model_buffer
    if _gesture_model_buffer is None:
        _gesture_model_buffer = load_model_buffer('./models/gesture_recognizer.task')
    return _gesture_model_buffer

def get_face_model_buffer():
    global _face_model_buffer
    if _face_model_buffer is None:
        _face_model_buffer = load_model_buffer('./models/face_landmarker.task')
    return _face_model_buffer

# Read video function using OpenCV
def read_video(file_path):
    """
    Read video file and convert to tensor
    Optimized to handle failures gracefully
    """
    try:
        # Use OpenCV to read video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {file_path}")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame)
            frames.append(frame_tensor)
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames read from video file: {file_path}")
            
        return torch.stack(frames)
    
    except Exception as e:
        logging.error(f"Error reading video {file_path}: {e}")
        raise


class KeypointExtractor:
    def __init__(self, use_gpu=True):
        """
        Initialize keypoint extractor with GPU delegate when available
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        logging.info(f"KeypointExtractor initialized with GPU: {self.use_gpu}, device: {self.device}")
        
    def extract_hand_landmarks(self, detection_result):
        """
        Extract hand landmarks from detection result
        """
        # Create tensor directly on the target device
        result = torch.zeros(len(hand_target_landmarks) * 2, 3, dtype=torch.float32, device=self.device) - 1
        
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = detection_result.handedness[i][0].category_name.lower()
            offset = 0 if hand_label != 'left' else len(hand_target_landmarks)
            
            for j, landmark in enumerate(hand_landmarks):
                if j < len(hand_target_landmarks):
                    result[j + offset][0] = landmark.x
                    result[j + offset][1] = landmark.y
                    result[j + offset][2] = landmark.z
        
        return result
    
    def extract_pose_landmarks(self, detection_result):
        """
        Extract pose landmarks from detection result
        """
        # Create tensor directly on the target device
        result = torch.zeros(len(pose_target_landmarks), 3, dtype=torch.float32, device=self.device) - 1
        
        if detection_result.pose_landmarks:
            for i, idx in enumerate(pose_target_landmarks):
                landmark = detection_result.pose_landmarks.landmark[idx]
                result[i][0] = landmark.x
                result[i][1] = landmark.y
                result[i][2] = landmark.z
        
        return result
    
    def extract_face_landmarks(self, detection_result):
        """
        Extract face landmarks from detection result
        """
        # Create tensor directly on the target device
        result = torch.zeros(len(face_target_landmarks), 3, dtype=torch.float32, device=self.device) - 1
        
        face_landmarks_list = detection_result.face_landmarks
        if face_landmarks_list and len(face_landmarks_list) > 0:
            face_landmarks = face_landmarks_list[0]
            for i, landmark in enumerate(face_landmarks):
                if i < len(face_target_landmarks):
                    result[i][0] = landmark.x
                    result[i][1] = landmark.y
                    result[i][2] = landmark.z
        
        return result

    def extract_fast(self, video, fps=24):
        """
        Extract keypoints in a highly optimized way using GPU delegate
        """
        # Keep video on the original device, no need to move to CPU
        num_frames = len(video)
        frame_indices = list(range(0, num_frames, 1))  # Process every frame
        h, w = video.shape[2], video.shape[3]
        
        # Preallocate result tensor on the target device
        results = torch.zeros((len(frame_indices), 
                          len(hand_target_landmarks) * 2 + len(face_target_landmarks) + len(pose_target_landmarks), 
                          3), dtype=torch.float32, device=self.device)
        
        # Get model buffers
        gesture_model_buffer = get_gesture_model_buffer()
        face_model_buffer = get_face_model_buffer()
        
        # Configure the GPU delegate if available
        delegate = BaseOptions.Delegate.GPU if self.use_gpu else BaseOptions.Delegate.CPU
        
        # Create model options with GPU delegate
        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_buffer=gesture_model_buffer,
                delegate=delegate  # Use GPU delegate
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=face_model_buffer,
                delegate=delegate  # Use GPU delegate
            ),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1
        )
        
        # Create models once for reuse
        face_landmarker = FaceLandmarker.create_from_options(face_options)
        
        # Note: MediaPipe Pose doesn't use the task API yet, so we use the traditional API
        # We'll use a lower complexity model for speed
        pose_landmarker = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=False
        )
        
        hand_landmarker = GestureRecognizer.create_from_options(gesture_options)
        
        try:
            # Process frames sequentially
            for idx, frame_idx in enumerate(frame_indices):
                timestamp = int(1000/fps * frame_idx)
                frame = video[frame_idx]
                
                # Convert to numpy
                # Use .cpu() only if tensor is on GPU - mediapipe requires numpy arrays
                if frame.device.type == 'cuda':
                    frame_np = frame.cpu().permute(1,2,0).mul(255).to(torch.uint8).numpy()
                else:
                    frame_np = frame.permute(1,2,0).mul(255).to(torch.uint8).numpy()
                
                # Process with pose landmarker
                image_rgb = frame_np.copy()  # No need to convert RGB again
                image_rgb.flags.writeable = False
                pose_result = pose_landmarker.process(image_rgb)
                
                # Process with hand and face landmarkers
                image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
                hands_result = hand_landmarker.recognize_for_video(image_mp, timestamp)
                face_result = face_landmarker.detect_for_video(image_mp, timestamp)
                
                # Extract landmarks into preallocated tensor
                hand_result = self.extract_hand_landmarks(hands_result)
                face_result_landmarks = self.extract_face_landmarks(face_result)
                pose_result_landmarks = self.extract_pose_landmarks(pose_result)
                
                # Concatenate into results
                results[idx] = torch.cat([hand_result, face_result_landmarks, pose_result_landmarks], dim=0)
        
        finally:
            # Clean up resources
            face_landmarker.close()
            pose_landmarker.close()
            hand_landmarker.close()
        
        # Scale by dimensions
        scale_tensor = torch.tensor([w, h, 1], dtype=torch.float32, device=self.device)
        return results * scale_tensor
    
    def extract_fast_parallel(self, video, fps=24):
        """
        Extract keypoints using parallel processing with GPU acceleration
        """
        # Process every frame for better accuracy
        stride = 1
        selected_indices = list(range(0, len(video), stride))
        video_subset = video[selected_indices]
        
        # Use appropriate number of workers
        num_workers = min(4, mp_threading.cpu_count() - 1)
        
        # Chunk the video frames for parallel processing
        frames_per_worker = max(1, len(video_subset) // num_workers)
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * frames_per_worker
            end_idx = start_idx + frames_per_worker if i < num_workers - 1 else len(video_subset)
            if start_idx < end_idx:  # Ensure non-empty chunks
                chunks.append((video_subset[start_idx:end_idx], start_idx * stride, fps, 
                              video.shape[2], video.shape[3], self.use_gpu))
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(self._process_chunk, chunks))
        
        # Combine results and ensure they're on the target device
        if chunk_results:
            # Move each chunk result to the target device if needed
            device_results = [r.to(self.device) if r.device != self.device else r for r in chunk_results]
            result = torch.cat(device_results, dim=0)
            
            # If we used stride > 1, interpolate the missing frames
            if stride > 1:
                result = self._interpolate_frames(result, len(video), stride)
            
            return result
        else:
            # Return empty tensor if no chunks were processed
            return torch.zeros((0, len(hand_target_landmarks) * 2 + len(face_target_landmarks) + len(pose_target_landmarks), 3), 
                              dtype=torch.float32, device=self.device)

    def _interpolate_frames(self, keypoints, total_frames, stride):
        """
        Linear interpolation to fill in skipped frames
        """
        full_results = torch.zeros((total_frames, keypoints.shape[1], keypoints.shape[2]), 
                                  dtype=torch.float32, device=keypoints.device)
        
        # Copy existing keypoints
        for i, idx in enumerate(range(0, total_frames, stride)):
            if i < len(keypoints):
                full_results[idx] = keypoints[i]
        
        # Linear interpolation for missing frames (vectorized where possible)
        for idx in range(total_frames):
            if idx % stride == 0:
                continue  # Already filled
            
            # Find nearest filled frames
            prev_idx = (idx // stride) * stride
            next_idx = min(prev_idx + stride, total_frames - 1)
            
            if next_idx == prev_idx:
                full_results[idx] = full_results[prev_idx]
            else:
                # Weight for linear interpolation
                weight = (idx - prev_idx) / (next_idx - prev_idx)
                full_results[idx] = (1 - weight) * full_results[prev_idx] + weight * full_results[next_idx]
        
        return full_results
    
    @staticmethod
    def _process_chunk(chunk_data):
        """
        Process a chunk of video frames
        Static method for parallel processing
        """
        video_chunk, start_idx, fps, height, width, use_gpu = chunk_data
        
        # Configure the GPU delegate if available
        delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        device = torch.device('cuda' if use_gpu else 'cpu')
        
        # Get model buffers
        gesture_model_buffer = get_gesture_model_buffer()
        face_model_buffer = get_face_model_buffer()
        
        # Create model options with GPU delegate
        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_buffer=gesture_model_buffer,
                delegate=delegate  # Use GPU delegate
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=face_model_buffer,
                delegate=delegate  # Use GPU delegate
            ),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1
        )
        
        # Create models for this process
        face_landmarker = FaceLandmarker.create_from_options(face_options)
        pose_landmarker = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=False
        )
        hand_landmarker = GestureRecognizer.create_from_options(gesture_options)
        
        # Create a KeypointExtractor for this chunk
        extractor = KeypointExtractor(use_gpu=use_gpu)
        results = []
        
        try:
            for frame_idx, frame in enumerate(video_chunk):
                actual_idx = start_idx + frame_idx
                timestamp = int(1000/fps * actual_idx)
                
                # Convert to numpy - mediapipe requires numpy arrays
                if frame.device.type == 'cuda':
                    frame_np = frame.cpu().permute(1,2,0).mul(255).to(torch.uint8).numpy()
                else:
                    frame_np = frame.permute(1,2,0).mul(255).to(torch.uint8).numpy()
                
                # Process with all models
                image_rgb = frame_np.copy()  # No need to convert RGB again
                image_rgb.flags.writeable = False
                pose_result = pose_landmarker.process(image_rgb)
                
                image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
                hands_result = hand_landmarker.recognize_for_video(image_mp, timestamp)
                face_result = face_landmarker.detect_for_video(image_mp, timestamp)
                
                # Extract landmarks
                result = torch.cat([
                    extractor.extract_hand_landmarks(hands_result),
                    extractor.extract_face_landmarks(face_result),
                    extractor.extract_pose_landmarks(pose_result),
                ], dim=0)
                
                results.append(result)
        
        finally:
            # Clean up resources
            face_landmarker.close()
            pose_landmarker.close()
            hand_landmarker.close()
        
        # Stack results
        if results:
            results_tensor = torch.stack(results)
            # Scale by dimensions
            scale_tensor = torch.tensor([width, height, 1], dtype=torch.float32, device=device)
            return results_tensor * scale_tensor
        else:
            # Return empty tensor with correct shape if no results
            return torch.zeros((0, len(hand_target_landmarks) * 2 + len(face_target_landmarks) + len(pose_target_landmarks), 3), 
                              dtype=torch.float32, device=device)