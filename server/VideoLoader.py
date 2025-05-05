import logging
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import sys
from bisect import bisect_right 
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp_threading
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode

# Load models once at module level to avoid reloading
with open('./models/gesture_recognizer.task', "rb") as f:
    gesture_model_buffer = f.read()
    
with open("./models/face_landmarker.task", "rb") as f:
    face_model_buffer = f.read()

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=gesture_model_buffer),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

mp_pose = mp.solutions.pose

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=face_model_buffer),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1
)

hand_target_landmarks = list(range(21))
face_target_landmarks = list(range(478))
pose_target_landmarks = list(range(33))


def read_video(file_path):
    try:
        video, audio, info = read_video(file_path, pts_unit='sec')
        return video
    except Exception as e:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame)
            frames.append(frame_tensor)
        cap.release()
        return torch.stack(frames)


class KeypointExtractor:
    def extract_hand_landmarks(self, detection_result):
        result = torch.zeros(len(hand_target_landmarks) * 2, 3, dtype=torch.float32) - 1
        
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
        result = torch.zeros(len(pose_target_landmarks), 3, dtype=torch.float32) - 1
        
        if detection_result.pose_landmarks:
            for i, idx in enumerate(pose_target_landmarks):
                landmark = detection_result.pose_landmarks.landmark[idx]
                result[i][0] = landmark.x
                result[i][1] = landmark.y
                result[i][2] = landmark.z
        
        return result
    
    def extract_face_landmarks(self, detection_result):
        result = torch.zeros(len(face_target_landmarks), 3, dtype=torch.float32) - 1
        
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
        Highly optimized version using minimal frame skipping and reduced overhead
        """
        # Reduce frame skipping for better accuracy
        every_nth = 1
        num_frames = len(video)
        frame_indices = list(range(0, num_frames, every_nth))
        h, w = video.shape[2], video.shape[3]
        
        # Preallocate result tensor
        results = torch.zeros((len(frame_indices), 
                              len(hand_target_landmarks) * 2 + len(face_target_landmarks) + len(pose_target_landmarks), 
                              3), dtype=torch.float32)
        
        # Create models once for reuse
        face_landmarker = FaceLandmarker.create_from_options(face_options)
        pose_landmarker = mp_pose.Pose(
            min_detection_confidence=0.5,  # Reduced for speed
            min_tracking_confidence=0.3,   # Reduced for speed
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=False
        )
        hand_landmarker = GestureRecognizer.create_from_options(gesture_options)
        
        try:
            # Process frames sequentially for simplicity
            for idx, frame_idx in enumerate(frame_indices):
                timestamp = int(1000/fps * frame_idx)
                frame = video[frame_idx]
                
                # Convert to numpy once
                frame_np = frame.permute(1,2,0).mul(255).to(torch.uint8).numpy()
                
                # Process with pose landmarker (most efficient path)
                image_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
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
        scale_tensor = torch.tensor([w, h, 1], dtype=torch.float32)
        return results * scale_tensor
    
    def extract_fast_parallel(self, video, fps=24):
        """
        Parallel processing version with optimized chunking
        """
        num_cores = mp_threading.cpu_count() - 1  # Leave one core free
        chunk_size = max(4, len(video) // num_cores)
        
        # Split video into chunks
        chunks = []
        for i in range(0, len(video), chunk_size):
            chunks.append((video[i:i+chunk_size], i, fps, video.shape[2], video.shape[3]))
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            chunk_results = [f.result() for f in futures]
        
        # Combine results
        return torch.cat(chunk_results, dim=0)
    
    @staticmethod
    def _process_chunk(chunk_data):
        """
        Static method for processing chunks in parallel
        """
        video_chunk, start_idx, fps, height, width = chunk_data
        
        # Create models for this process
        face_landmarker = FaceLandmarker.create_from_options(face_options)
        pose_landmarker = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=False
        )
        hand_landmarker = GestureRecognizer.create_from_options(gesture_options)
        
        extractor = KeypointExtractor()
        results = []
        
        try:
            for frame_idx, frame in enumerate(video_chunk):
                actual_idx = start_idx + frame_idx
                timestamp = int(1000/fps * actual_idx)
                
                # Convert to numpy once
                frame_np = frame.permute(1,2,0).mul(255).to(torch.uint8).numpy()
                
                # Process with all models
                image_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
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
        
        # Convert to tensor and scale
        results = torch.stack(results)
        scale_tensor = torch.tensor([width, height, 1], dtype=torch.float32)
        return results * scale_tensor