import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import sys

from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=open("face_landmarker.task", "rb").read()),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes = True,
    num_faces = 1)

hand_target_landmarks = list(range(21))
pose_target_landmarks = list(range(23))
face_target_landmarks = list(range(52))


keys_list = ['timestamp']
for i in hand_target_landmarks:
    keys_list.append(f'left_{i}_x')
    keys_list.append(f'left_{i}_y')
    keys_list.append(f'left_{i}_z')

    keys_list.append(f'right_{i}_x')
    keys_list.append(f'right_{i}_y')
    keys_list.append(f'right_{i}_z')

for i in pose_target_landmarks:
    keys_list.append(f'pose_{i}_x')
    keys_list.append(f'pose_{i}_y')
    keys_list.append(f'pose_{i}_z')

keys_list += ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

hand_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=0)
pose_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=0)




class VideoExtractor:

    def __init__(self, path):
        self.path = path
        self.landmark_data = []


    def get_data(self):
        return self.landmark_data
    

    def save(self, filename):
        # Convert data to structured NumPy array
        dtype = [(key, 'float32') for key in keys_list]
        structured_array = np.full((len(self.landmark_data),), np.nan, dtype=dtype)
    
        for i, frame in enumerate(self.landmark_data):
            for key in frame:
                structured_array[key][i] = frame[key]
    
        np.savez_compressed(filename, data=structured_array)


    def extract_hand_landmarks(self, detection_result):
        result = {}
    
        if detection_result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.multi_hand_landmarks):
                hand_label = detection_result.multi_handedness[i].classification[0].label.lower()
                
                for idx in hand_target_landmarks:
                    landmark = hand_landmarks.landmark[idx]
                    result[f'{hand_label}_{idx}_x'] = landmark.x
                    result[f'{hand_label}_{idx}_y'] = landmark.x
                    result[f'{hand_label}_{idx}_z'] = landmark.x        
        return result

    
    def extract_pose_landmarks(self, detection_result):
        result = {}
        
        if detection_result.pose_landmarks:
            for idx in pose_target_landmarks:
    
                landmarks = detection_result.pose_landmarks.landmark[idx]
                result[f'pose_{idx}_x'] = landmarks.x
                result[f'pose_{idx}_y'] = landmarks.y
                result[f'pose_{idx}_z'] = landmarks.z
                
        return result
    

    def extract_face_blendshapes(self, detection_result):
        face_blendshapes_list = detection_result.face_blendshapes
        result = {}
        
        for idx in range(len(face_blendshapes_list)):
            face_blendshapes = face_blendshapes_list[idx]
    
            face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
            face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
            
            for num in face_target_landmarks:
                result[face_blendshapes_names[num]] = face_blendshapes_scores[num]
    
        return result


    def crop_to_square(self, frame):
        # Get the dimensions of the frame
        height, width, _ = frame.shape
    
        # Find the size of the square (the smaller of the width or height)
        square_size = int(min(height, width))
    
        # Calculate the cropping coordinates to make the frame square
        start_x = (width - square_size) // 2
        start_y = (height - square_size) // 2
    
        # Crop the frame to a square (from the center)
        cropped_frame = frame[start_y:start_y + square_size, start_x:start_x + square_size,:]
        return cropped_frame



    def draw_landmarks(self, frame, hands_result, face_result, pose_result):
        
        frame = np.copy(frame)
        
        if hands_result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, hand_style, hand_style)
    
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS, pose_style, pose_style)
    
    
        face_landmarks_list = face_result.face_landmarks
        
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            
            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp.solutions.drawing_styles
                  .get_default_face_mesh_iris_connections_style())
    
        
        return frame
    

    def extract(self, visualize_output=None):
        cap = cv2.VideoCapture(self.path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
        if visualize_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(visualize_output, fourcc, fps, width, height)
        
        with FaceLandmarker.create_from_options(face_options) as face_landmarker, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.4, model_complexity=0) as pose_landmarker, \
            mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.3, model_complexity=1) as hand_landmarker:
      
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

    
                #frame = self.crop_to_square(frame)
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # Video timestamp in milliseconds
                
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                hands_result = hand_landmarker.process(image)
                pose_result = pose_landmarker.process(image)
                
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
                face_result = face_landmarker.detect_for_video(image, timestamp)
    
    
                frame_data = {'timestamp' : timestamp}
    
                frame_data.update(self.extract_hand_landmarks(hands_result))
                frame_data.update(self.extract_pose_landmarks(pose_result))
                frame_data.update(self.extract_face_blendshapes(face_result))
    
                if visualize_output:
                    frame = self.draw_landmarks(frame, hands_result, face_result, pose_result)
                    out.write(frame)
        
                self.landmark_data.append(frame_data)
    
        cap.release()
        if visualize_output:
            out.release()
            




        
        
