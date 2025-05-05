import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import sys
from bisect import bisect_right 

from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult


VisionRunningMode = mp.tasks.vision.RunningMode


gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=open('./models/gesture_recognizer.task', "rb").read()),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2)


mp_pose = mp.solutions.pose


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=open("./models/face_landmarker.task", "rb").read()),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes = True,
    num_faces = 1)

hand_target_landmarks = list(range(21))
face_target_landmarks = list(range(478))
pose_target_landmarks = list(range(33))


#face_target_landmarks = [291, 267, 37, 61, 84, 314, 310, 13, 80, 14] + [152]
#pose_target_landmarks = [2, 5, 7, 8, 11, 12, 13, 14, 15, 16]


import torch


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
        result = torch.zeros(len(hand_target_landmarks) * 2, 3)-1
                
        
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):

            hand_label = detection_result.handedness[i][0].category_name.lower()
            for j, idx in enumerate(hand_target_landmarks):
                
                i = j
                if hand_label == 'left':
                    i += len(hand_target_landmarks)
                
                landmark = hand_landmarks[idx]
                result[i][0] = landmark.x
                result[i][1] = landmark.y   
                result[i][2] = landmark.z   
        
        return result

    
    def extract_pose_landmarks(self, detection_result):
        result = torch.zeros(len(pose_target_landmarks), 3)-1
        
        if detection_result.pose_landmarks:
            for i, idx in enumerate(pose_target_landmarks):
    
                landmark = detection_result.pose_landmarks.landmark[idx]
                result[i][0] = landmark.x
                result[i][1] = landmark.y   
                result[i][2] = landmark.z
                
        return result
    

    def extract_face_landmarks(self, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        face_landmarks = []
        result = torch.zeros(len(face_target_landmarks), 3)-1

        
        for i in range(len(face_landmarks_list)):
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    
            for j, landmark in enumerate(face_landmarks_list[i]):
                face_landmarks.append((landmark.x, landmark.y, landmark.z))
            break

        if len(face_landmarks) > 400:
            for i, idx in enumerate(face_target_landmarks):
                result[i][0] = face_landmarks[idx][0]
                result[i][1] = face_landmarks[idx][1]
                result[i][2] = face_landmarks[idx][2]
                
    
        return result
    
    def _process_frame_batch(self, frame_batch, frame_indices, fps, pose_landmarker, face_landmarker, hand_landmarker):
        """Process a batch of frames together"""
        batch_results = []
        
        for frame_idx in frame_indices:
            frame = frame_batch[frame_idx]
            timestamp = int(1000/fps * frame_idx)
            
            # Convert tensor to numpy array
            frame_np = np.array(frame.permute(1,2,0) * 255).astype(np.uint8)
            
            # Process with pose landmarker
            image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            pose_result = pose_landmarker.process(image)
            
            # Process with hand and face landmarkers
            image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
            hands_result = hand_landmarker.recognize_for_video(image_mp, timestamp)
            face_result = face_landmarker.detect_for_video(image_mp, timestamp)
            
            # Extract landmarks
            result = torch.cat([
                self.extract_hand_landmarks(hands_result),
                self.extract_face_landmarks(face_result),
                self.extract_pose_landmarks(pose_result),
            ], dim=0)
            
            batch_results.append(result)
        
        return batch_results

    def extract_slow(self, video, fps=24):
        """
        Optimized extraction using batch processing and threading
        """
        print("cowabunga=============")
        results = []
        
        with FaceLandmarker.create_from_options(face_options) as face_landmarker, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.4, model_complexity=0) as pose_landmarker, \
            GestureRecognizer.create_from_options(gesture_options) as hand_landmarker:
            
            # Process frames in batches
            batch_size = 8  # Adjust based on memory constraints
            num_frames = len(video)
            
            for i in range(0, num_frames, batch_size):
                frame_batch = video[i:i + batch_size]
                frame_indices = range(i, min(i + batch_size, num_frames))
                
                # Process batch
                batch_results = self._process_frame_batch(
                    frame_batch, frame_indices, fps, 
                    pose_landmarker, face_landmarker, hand_landmarker
                )
                
                results.extend(batch_results)
        
        # Convert to tensor
        results = torch.stack(results)
        h, w = video.shape[2], video.shape[3]
        return results * torch.tensor([w, h, 1], dtype=results.dtype)
    
    def extract(self, video, fps=24):
        """
        Fastest version using frame skipping and parallel processing
        """
        print("cowabunga=============D")
        # Skip half the frames for faster processing (can be adjusted)
        every_nth = 2
        frame_indices = list(range(0, len(video), every_nth))
        
        # Process frames in parallel
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        results = [None] * len(frame_indices)
        lock = threading.Lock()
        
        def process_frame(idx, frame_idx):
            timestamp = int(1000/fps * frame_idx)
            frame = video[frame_idx]
            
            # Convert tensor to numpy array
            frame_np = np.array(frame.permute(1,2,0) * 255).astype(np.uint8)
            
            # Create separate instances for each thread
            with lock:
                with FaceLandmarker.create_from_options(face_options) as face_landmarker, \
                    mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.4, model_complexity=0) as pose_landmarker, \
                    GestureRecognizer.create_from_options(gesture_options) as hand_landmarker:
                    
                    # Process with pose landmarker
                    image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    pose_result = pose_landmarker.process(image)
                    
                    # Process with hand and face landmarkers
                    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
                    hands_result = hand_landmarker.recognize_for_video(image_mp, timestamp)
                    face_result = face_landmarker.detect_for_video(image_mp, timestamp)
                    
                    # Extract landmarks
                    result = torch.cat([
                        self.extract_hand_landmarks(hands_result),
                        self.extract_face_landmarks(face_result),
                        self.extract_pose_landmarks(pose_result),
                    ], dim=0)
                    
                    results[idx] = result
        
        # Use thread pool to process frames
        with ThreadPoolExecutor(max_workers=4) as executor:
            for idx, frame_idx in enumerate(frame_indices):
                executor.submit(process_frame, idx, frame_idx)
        
        # Interpolate skipped frames
        if every_nth > 1:
            full_results = []
            for i in range(len(video)):
                if i in frame_indices:
                    idx = frame_indices.index(i)
                    full_results.append(results[idx])
                else:
                    # Linear interpolation between adjacent frames
                    prev_idx = max(0, bisect_right(frame_indices, i) - 1)
                    next_idx = min(len(frame_indices) - 1, bisect_right(frame_indices, i))
                    
                    if prev_idx != next_idx:
                        prev_frame = results[prev_idx]
                        next_frame = results[next_idx]
                        ratio = (i - frame_indices[prev_idx]) / (frame_indices[next_idx] - frame_indices[prev_idx])
                        interpolated = prev_frame * (1 - ratio) + next_frame * ratio
                        full_results.append(interpolated)
                    else:
                        full_results.append(results[prev_idx])
            
            results = full_results
        
        # Convert to tensor
        results = torch.stack(results)
        h, w = video.shape[2], video.shape[3]
        return results * torch.tensor([w, h, 1], dtype=results.dtype)