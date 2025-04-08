from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# MediaPipe setup (similar to your original code)
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)

recognizer = GestureRecognizer.create_from_options(options)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = recognizer.recognize(mp_image)

    if results.gestures and len(results.gestures) > 0:
        detected_gesture = results.gestures[0][0].category_name
        return jsonify({'gesture': detected_gesture})
    else:
        return jsonify({'gesture': 'No gesture detected'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)