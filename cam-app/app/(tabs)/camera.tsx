import { SetStateAction, useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { AntDesign } from "@expo/vector-icons";
import * as FileSystem from "expo-file-system";
import * as Speech from "expo-speech";
import { Colors } from "@/constants/Colors";
import { Audio } from "expo-av";
import { SpeechToText } from "@/components/SpeechToText";

export default function CameraComponent() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [gesture, setGesture] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recognizedText, setRecognizedText] = useState("");
  const [transcriptionUri, setTranscriptionUri] = useState<string | null>(null);

  const cameraRef = useRef<CameraView | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    console.log("[DEBUG] Checking camera permissions...");
  }, []);

  if (!permission) {
    console.log("[DEBUG] Camera permissions are still loading...");
    return <View />;
  }
  if (!permission.granted) {
    console.log("[DEBUG] Camera permissions not granted");
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: "center" }}>
          We need your permission to show the camera
        </Text>
        <TouchableOpacity onPress={requestPermission} style={styles.button}>
          <Text style={styles.text}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    console.log("[DEBUG] Toggling camera facing...");
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  const startRealTimeDetection = () => {
    console.log("[DEBUG] Starting real-time gesture detection...");
    setIsDetecting(true);
    intervalRef.current = setInterval(async () => {
      if (cameraRef.current) {
        console.log("[DEBUG] Capturing frame...");
        const options = { quality: 0.5, base64: true, exif: false };
        try {
          const photo = await cameraRef.current.takePictureAsync(options);
          console.log(
            `[DEBUG] Frame captured. Size: ${photo?.width}x${photo?.height}`
          );
          sendFrameToServer(photo);
        } catch (error) {
          console.error("[ERROR] Failed to capture frame:", error);
        }
      }
    }, 300);
  };

  const stopRealTimeDetection = () => {
    console.log("[DEBUG] Stopping real-time gesture detection...");
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const sendFrameToServer = async (photo: any) => {
    try {
      console.log("[DEBUG] Preparing frame for upload...");
      let formData = new FormData();

      const response = await fetch(photo.uri);
      const blob = await response.blob();

      formData.append("file", blob, "frame.jpg");

      console.log("[DEBUG] Sending frame to server...");

      let responseFetch = await fetch(
        "http://127.0.0.1:8000/recognize-gesture/",
        {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
        }
      );

      console.log(`[DEBUG] Server response status: ${responseFetch.status}`);

      if (!responseFetch.ok) {
        const errorText = await responseFetch.text();
        console.error("[ERROR] Server response:", errorText);
        throw new Error(errorText);
      }

      let data = await responseFetch.json();
      console.log(`[DEBUG] Received gesture response: ${data.gesture}`);
      if (data.gesture !== "None") {
        Speech.speak(data.gesture);
      }
      setGesture(data.gesture);
    } catch (error) {
      console.error("[ERROR] Error sending frame:", error);
    }
  };

  const startRecording = async () => {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        console.error("Microphone permission not granted");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
    }
  };

  const stopRecording = async () => {
    try {
      if (!recording) return;

      await recording.stopAndUnloadAsync();
      setIsRecording(false);
      const uri = recording.getURI();
      setRecording(null);

      if (uri) {
        // Pass the recorded URI to the SpeechToText component
        setTranscriptionUri(uri);
      }
    } catch (err) {
      console.error("Failed to stop recording:", err);
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
            <AntDesign name="retweet" size={44} color="white" />
          </TouchableOpacity>
          {!isDetecting ? (
            <TouchableOpacity
              style={styles.button}
              onPress={startRealTimeDetection}
            >
              <AntDesign name="playcircleo" size={44} color="white" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={styles.button}
              onPress={stopRealTimeDetection}
            >
              <AntDesign name="pausecircleo" size={44} color="white" />
            </TouchableOpacity>
          )}
          {!isRecording ? (
            <TouchableOpacity style={styles.button} onPress={startRecording}>
              <Text style={styles.buttonText}>Record{"\n"}Audio</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.button} onPress={stopRecording}>
              <Text style={styles.buttonText}>Stop{"\n"}Recording</Text>
            </TouchableOpacity>
          )}
        </View>
      </CameraView>

      {transcriptionUri && (
        <SpeechToText
          audioUri={transcriptionUri}
          onTranscriptionComplete={(text: SetStateAction<string>) => {
            setRecognizedText(text);
            setTranscriptionUri(null);
          }}
        />
      )}

      {recognizedText ? (
        <View style={styles.textContainer}>
          <Text style={styles.recognizedText}>
            Recognized: {recognizedText}
          </Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: "absolute",
    top: 50,
    left: 50,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    padding: 10,
    borderRadius: 10,
  },
  gestureText: {
    fontSize: 24,
    color: "white",
    fontWeight: "bold",
  },
  buttonContainer: {
    flexDirection: "row",
    backgroundColor: "transparent",
    justifyContent: "center",
    alignItems: "flex-end",
    marginBottom: 20,
  },
  button: {
    width: 100,
    height: 80,
    marginHorizontal: 10,
    backgroundColor: "rgba(40, 40, 40, 0.8)",
    borderRadius: 40,
    borderWidth: 2,
    borderColor: Colors.light.tint,
    shadowColor: Colors.light.tint,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
  textContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    padding: 10,
  },
  recognizedText: {
    color: "white",
    textAlign: "center",
    fontSize: 16,
  },
  text: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
});
