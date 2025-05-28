import cv2
import mediapipe as mp
import torch
import multiprocessing as mp_threading
from torchvision.io import read_video as rv
import numpy as np
from functools import lru_cache
import os
import gc
import threading

# Force CPU-only execution for MediaPipe
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_v'] = '-1'
os.environ['GLOG_minloglevel'] = '3'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode

# Load models once at module level
with open('./models/gesture_recognizer.task', "rb") as f:
    gesture_model_buffer = f.read()
    
with open("./models/face_landmarker.task", "rb") as f:
    face_model_buffer = f.read()

# CPU-optimized options with minimal resource usage
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_buffer=gesture_model_buffer,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_buffer=face_model_buffer,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    num_faces=1
)

mp_pose = mp.solutions.pose

hand_target_landmarks = list(range(21))
face_target_landmarks = list(range(478))
pose_target_landmarks = list(range(33))

def read_video(file_path):
    try:
        video, audio, info = rv(file_path, pts_unit='sec')
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
    def __init__(self):
        self._lock = threading.Lock()  # Thread safety
        
    def extract_hand_landmarks(self, detection_result):
        result = torch.zeros(len(hand_target_landmarks) * 2, 3, dtype=torch.float32)
        result.fill_(-1)
        
        if not detection_result.hand_landmarks:
            return result
            
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            if i >= len(detection_result.handedness):
                continue
                
            hand_label = detection_result.handedness[i][0].category_name.lower()
            offset = 0 if hand_label != 'left' else len(hand_target_landmarks)
            
            for j, landmark in enumerate(hand_landmarks):
                if j < len(hand_target_landmarks):
                    idx = j + offset
                    if idx < len(result):
                        result[idx][0] = landmark.x
                        result[idx][1] = landmark.y
                        result[idx][2] = landmark.z
        
        return result
    
    def extract_pose_landmarks(self, detection_result):
        result = torch.zeros(len(pose_target_landmarks), 3, dtype=torch.float32)
        result.fill_(-1)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks.landmark
            for i, idx in enumerate(pose_target_landmarks):
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    result[i][0] = landmark.x
                    result[i][1] = landmark.y
                    result[i][2] = landmark.z
        
        return result
    
    def extract_face_landmarks(self, detection_result):
        result = torch.zeros(len(face_target_landmarks), 3, dtype=torch.float32)
        result.fill_(-1)
        
        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            face_landmarks = detection_result.face_landmarks[0]
            for i, landmark in enumerate(face_landmarks):
                if i < len(face_target_landmarks):
                    result[i][0] = landmark.x
                    result[i][1] = landmark.y
                    result[i][2] = landmark.z
        
        return result

    def extract_fast_parallel(self, video, fps=24):
        """
        Safe implementation that avoids memory corruption
        """
        with self._lock:  # Thread safety
            return self._extract_safe_sequential(video, fps)
    
    def extract_safe_parallel(self, video, fps=24):
        """
        Completely safe sequential processing - no parallel execution
        """
        return self._extract_safe_sequential(video, fps)
    
    def extract_sequential_safe(self, video, fps=24):
        """
        Ultra-safe sequential processing
        """
        return self._extract_safe_sequential(video, fps)

    def _extract_safe_sequential(self, video, fps=24):
        """
        Single-threaded keypoint extraction to prevent malloc corruption
        """
        num_frames = len(video)
        stride = 1
        # # Smart frame selection based on video length
        # if num_frames > 150:
        #     stride = 4  # Process every 4th frame for very long videos
        # elif num_frames > 100:
        #     stride = 3  # Process every 3rd frame
        # elif num_frames > 50:
        #     stride = 2  # Process every 2nd frame
        # else:
        #     stride = 1  # Process all frames for short videos
            
        selected_indices = list(range(0, num_frames, stride))
        video_subset = video[selected_indices]
        
        print(f"Processing {len(video_subset)} frames out of {num_frames} (stride={stride})")
        
        # Initialize MediaPipe models once
        face_landmarker = None
        pose_landmarker = None
        hand_landmarker = None
        
        try:
            # Create models with minimal resource usage
            face_landmarker = FaceLandmarker.create_from_options(face_options)
            pose_landmarker = mp_pose.Pose(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.2,
                model_complexity=0,  # Fastest model
                enable_segmentation=False,
                smooth_landmarks=False
            )
            hand_landmarker = GestureRecognizer.create_from_options(gesture_options)
            
            results = []
            height, width = video.shape[2], video.shape[3]
            
            # Process frames one by one
            for frame_idx, frame in enumerate(video_subset):
                try:
                    timestamp = int(1000/fps * frame_idx)
                    
                    # Safe frame conversion
                    if frame.is_cuda:
                        frame = frame.cpu()
                    
                    # Convert to numpy safely
                    frame_np = frame.permute(1, 2, 0).numpy()
                    if frame_np.dtype != np.uint8:
                        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                    
                    # Ensure contiguous memory layout
                    frame_np = np.ascontiguousarray(frame_np)
                    
                    # Process with pose (most stable)
                    image_rgb = frame_np.copy()
                    image_rgb.flags.writeable = False
                    pose_result = pose_landmarker.process(image_rgb)
                    
                    # Process with MediaPipe tasks
                    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
                    hands_result = hand_landmarker.recognize_for_video(image_mp, timestamp)
                    face_result = face_landmarker.detect_for_video(image_mp, timestamp)
                    
                    # Extract landmarks
                    hand_landmarks = self.extract_hand_landmarks(hands_result)
                    face_landmarks = self.extract_face_landmarks(face_result)
                    pose_landmarks = self.extract_pose_landmarks(pose_result)
                    
                    # Combine landmarks
                    combined_result = torch.cat([
                        hand_landmarks,
                        face_landmarks,
                        pose_landmarks
                    ], dim=0)
                    
                    results.append(combined_result)
                    
                    # Clear intermediate variables
                    del frame_np, image_rgb, image_mp
                    del hands_result, face_result, pose_result
                    del hand_landmarks, face_landmarks, pose_landmarks
                    
                    # Periodic garbage collection for long videos
                    if frame_idx % 20 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    # Add empty result to maintain frame consistency
                    total_landmarks = (len(hand_target_landmarks) * 2 + 
                                     len(face_target_landmarks) + 
                                     len(pose_target_landmarks))
                    empty_result = torch.zeros(total_landmarks, 3, dtype=torch.float32) - 1
                    results.append(empty_result)
        
        except Exception as e:
            print(f"Critical error in keypoint extraction: {e}")
            # Return empty results
            total_landmarks = (len(hand_target_landmarks) * 2 + 
                             len(face_target_landmarks) + 
                             len(pose_target_landmarks))
            num_frames_to_process = len(video_subset)
            return torch.zeros((num_frames_to_process, total_landmarks, 3), dtype=torch.float32) - 1
        
        finally:
            # Clean up MediaPipe resources
            if face_landmarker:
                try:
                    face_landmarker.close()
                except:
                    pass
            if pose_landmarker:
                try:
                    pose_landmarker.close()
                except:
                    pass
            if hand_landmarker:
                try:
                    hand_landmarker.close()
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
        
        if not results:
            print("No results generated")
            total_landmarks = (len(hand_target_landmarks) * 2 + 
                             len(face_target_landmarks) + 
                             len(pose_target_landmarks))
            return torch.zeros((1, total_landmarks, 3), dtype=torch.float32) - 1
        
        # Convert to tensor and scale
        results_tensor = torch.stack(results)
        scale_tensor = torch.tensor([width, height, 1], dtype=torch.float32)
        scaled_results = results_tensor * scale_tensor
        
        # Interpolate if we used stride > 1
        if stride > 1:
            final_results = self._safe_interpolation(scaled_results, selected_indices, num_frames)
        else:
            final_results = scaled_results
        
        print(f"Keypoint extraction completed. Final shape: {final_results.shape}")
        return final_results

    def _safe_interpolation(self, keypoints, selected_indices, total_frames):
        """
        Memory-safe interpolation with OpenMP parallelization
        """
        if len(selected_indices) == total_frames:
            return keypoints
        
        # Calculate stride
        stride = selected_indices[1] - selected_indices[0] if len(selected_indices) > 1 else 1
        
        try:
            # Use PyTorch's vectorized operations (automatically uses OpenMP)
            device = keypoints.device
            dtype = keypoints.dtype
            
            full_results = torch.zeros((total_frames, keypoints.shape[1], keypoints.shape[2]), 
                                     dtype=dtype, device=device)
            
            # Vectorized copying of existing keypoints
            valid_keypoint_indices = torch.arange(len(keypoints), device=device)
            frame_indices = torch.tensor(selected_indices, device=device)
            
            # Ensure we don't exceed bounds
            valid_mask = (valid_keypoint_indices < len(keypoints)) & (frame_indices < total_frames)
            valid_keypoint_idx = valid_keypoint_indices[valid_mask]
            valid_frame_idx = frame_indices[valid_mask]
            
            if len(valid_keypoint_idx) > 0:
                full_results[valid_frame_idx] = keypoints[valid_keypoint_idx]
            
            # Parallel interpolation for missing frames
            all_indices = torch.arange(total_frames, device=device)
            selected_set = set(selected_indices)
            missing_indices = torch.tensor([i for i in range(total_frames) if i not in selected_set], 
                                         device=device)
            
            if len(missing_indices) > 0:
                # Vectorized interpolation
                for idx in missing_indices:
                    idx_val = idx.item()
                    
                    # Find nearest keypoints
                    prev_indices = [i for i in selected_indices if i < idx_val]
                    next_indices = [i for i in selected_indices if i > idx_val]
                    
                    if prev_indices and next_indices:
                        prev_idx = max(prev_indices)
                        next_idx = min(next_indices)
                        
                        weight = (idx_val - prev_idx) / (next_idx - prev_idx)
                        full_results[idx] = ((1 - weight) * full_results[prev_idx] + 
                                           weight * full_results[next_idx])
                    elif prev_indices:
                        full_results[idx] = full_results[max(prev_indices)]
                    elif next_indices:
                        full_results[idx] = full_results[min(next_indices)]
            
            return full_results
            
        except Exception as e:
            print(f"Parallel interpolation error: {e}, falling back to simple method")
            return self._simple_interpolation_fallback(keypoints, selected_indices, total_frames)

    def _simple_interpolation_fallback(self, keypoints, selected_indices, total_frames):
        """
        Simple fallback interpolation method
        """
        full_results = torch.zeros((total_frames, keypoints.shape[1], keypoints.shape[2]), 
                                 dtype=keypoints.dtype, device=keypoints.device)
        full_results.fill_(-1)
        
        # Copy existing keypoints
        for i, idx in enumerate(selected_indices):
            if i < len(keypoints) and idx < total_frames:
                full_results[idx] = keypoints[i]
        
        # Simple linear interpolation
        selected_set = set(selected_indices)
        for idx in range(total_frames):
            if idx in selected_set:
                continue
            
            prev_indices = [i for i in selected_indices if i < idx]
            next_indices = [i for i in selected_indices if i > idx]
            
            if prev_indices and next_indices:
                prev_idx = max(prev_indices)
                next_idx = min(next_indices)
                weight = (idx - prev_idx) / (next_idx - prev_idx)
                full_results[idx] = ((1 - weight) * full_results[prev_idx] + 
                                   weight * full_results[next_idx])
            elif prev_indices:
                full_results[idx] = full_results[max(prev_indices)]
            elif next_indices:
                full_results[idx] = full_results[min(next_indices)]
        
        return full_results