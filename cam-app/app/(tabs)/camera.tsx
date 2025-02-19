import { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import { AntDesign } from '@expo/vector-icons';

export default function CameraComponent() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [gesture, setGesture] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);

  const cameraRef = useRef<CameraView | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    console.log('[DEBUG] Checking camera permissions...');
  }, []);

  if (!permission) {
    console.log('[DEBUG] Camera permissions are still loading...');
    return <View />;
  }
  if (!permission.granted) {
    console.log('[DEBUG] Camera permissions not granted');
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.button}>
          <Text style={styles.text}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    console.log('[DEBUG] Toggling camera facing...');
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }

  const startRealTimeDetection = () => {
    console.log('[DEBUG] Starting real-time gesture detection...');
    setIsDetecting(true);
    intervalRef.current = setInterval(async () => {
      if (cameraRef.current) {
        console.log('[DEBUG] Capturing frame...');
        const options = { quality: 0.5, base64: true, exif: false };
        try {
          const photo = await cameraRef.current.takePictureAsync(options);
          console.log(`[DEBUG] Frame captured. Size: ${photo?.width}x${photo?.height}`);
          sendFrameToServer(photo);
        } catch (error) {
          console.error('[ERROR] Failed to capture frame:', error);
        }
      }
    }, 500); // Adjust interval based on server response speed
  };

  const stopRealTimeDetection = () => {
    console.log('[DEBUG] Stopping real-time gesture detection...');
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const sendFrameToServer = async (photo: any) => {
    try {
      console.log('[DEBUG] Preparing frame for upload...');
      let formData = new FormData();
      const photoBlob = await (await fetch(photo.uri)).blob();
      formData.append('file', photoBlob, 'frame.jpg');

      console.log('[DEBUG] Sending frame to server...');
      let response = await fetch('http://127.0.0.1:8000/recognize-gesture/', {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log(`[DEBUG] Server response status: ${response.status}`);
      let data = await response.json();
      console.log(`[DEBUG] Received gesture response: ${data.gesture}`);
      setGesture(data.gesture);
    } catch (error) {
      console.error('[ERROR] Error sending frame:', error);
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
        {gesture && (
          <View style={styles.overlay}>
            <Text style={styles.gestureText}>Gesture: {gesture}</Text>
          </View>
        )}

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <AntDesign name="retweet" size={44} color="black" />
          </TouchableOpacity>
          {!isDetecting ? (
            <TouchableOpacity style={styles.button} onPress={startRealTimeDetection}>
              <AntDesign name="playcircleo" size={44} color="black" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.button} onPress={stopRealTimeDetection}>
              <AntDesign name="pausecircleo" size={44} color="black" />
            </TouchableOpacity>
          )}
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 50,
    left: 50,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 10,
    borderRadius: 10,
  },
  gestureText: {
    fontSize: 24,
    color: 'white',
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'flex-end',
    marginBottom: 20,
  },
  button: {
    padding: 15,
    marginHorizontal: 10,
    backgroundColor: 'gray',
    borderRadius: 10,
  },
  text: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
});
