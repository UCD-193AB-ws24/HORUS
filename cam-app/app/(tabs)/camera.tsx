import { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import { SpeechToText } from "@/components/SpeechToText";
import { AntDesign } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';

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
    }, 200); // Adjust interval based on server response speed
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

        const response = await fetch(photo.uri);
        const blob = await response.blob();

        formData.append('file', blob, 'frame.jpg');

        console.log('[DEBUG] Sending frame to server...');

        let responseFetch = await fetch('http://127.0.0.1:8000/recognize-gesture/', {
            method: 'POST',
            body: formData,
            headers: {
                "Accept": "application/json",
            },
        });

        console.log(`[DEBUG] Server response status: ${responseFetch.status}`);

        if (!responseFetch.ok) {
            const errorText = await responseFetch.text();
            console.error('[ERROR] Server response:', errorText);
            throw new Error(errorText);
        }

        let data = await responseFetch.json();
        console.log(`[DEBUG] Received gesture response: ${data.gesture}`);
        if (data.gesture != "None"){
          Speech.speak(data.gesture);
        }
        setGesture(data.gesture);

    } catch (error) {
        console.error('[ERROR] Error sending frame:', error);
    }
};


  const savePhotoToLocalFolder = async (photoUri: string) => {
    try {
      const photoBlob = await (await fetch(photoUri)).blob();
      const fileUri = `${FileSystem.documentDirectory}frame.jpg`;
  
      // Convert blob to base64
      const reader = new FileReader();
      reader.readAsDataURL(photoBlob);
      reader.onloadend = async () => {
        const base64data = reader.result as string;
        await FileSystem.writeAsStringAsync(fileUri, base64data.split(',')[1], {
          encoding: FileSystem.EncodingType.Base64,
        });
        console.log('Photo saved to:', fileUri);
      };
    } catch (error) {
      console.error('Error saving photo:', error);
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
      <SpeechToText/> 
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
