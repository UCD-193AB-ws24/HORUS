import { SetStateAction, useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { AntDesign } from "@expo/vector-icons";
import * as Speech from "expo-speech";
import { Colors } from "@/constants/Colors";
import { Audio } from "expo-av";
import { SpeechToText } from "@/components/SpeechToText";
import { HOSTNAME } from "@env";
import * as MediaLibrary from "expo-media-library";

export default function CameraComponent() {
  const [facing, setFacing] = useState<CameraType>("front"); // Set front camera as default for signing
  const [permission, requestPermission] = useCameraPermissions();
  const [recognizedWord, setRecognizedWord] = useState<string | null>(null);
  const [isVideoRecording, setIsVideoRecording] = useState(false);
  const [isAudioRecording, setIsAudioRecording] = useState(false);
  const [audioRecording, setAudioRecording] = useState<Audio.Recording | null>(null);
  const [recognizedText, setRecognizedText] = useState("");
  const [transcriptionUri, setTranscriptionUri] = useState<string | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  
  const cameraRef = useRef<CameraView | null>(null);

  useEffect(() => {
    console.log("[DEBUG] Checking camera permissions...");
  }, []);

  // Add recognized words to our sentence
  useEffect(() => {
    if (recognizedWord && recognizedWord !== "None") {
      setSentence(prev => [...prev, recognizedWord]);
    }
  }, [recognizedWord]);

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

  // Actually start video recording
  const startVideoRecording = async () => {
    try {
      console.log("[DEBUG] Starting actual video recording...");
      
      if (!cameraRef.current) {
        console.error("[ERROR] Camera reference is null");
        return;
      }
      
      setIsVideoRecording(true);
      
      // Using the recordAsync method with updated options
      // Note: quality is now set as a prop on CameraView, not in recordAsync options
      const videoRecordPromise = cameraRef.current.recordAsync({
        maxDuration: 5, // Maximum duration in seconds
        mirror: true      // Mirror for front camera (selfie view)
      });
      
      // Set up the promise resolution
      const recordedVideo = await videoRecordPromise;
      console.log("[DEBUG] Video recording completed:", recordedVideo.uri);
      setVideoUri(recordedVideo.uri);
      
      // Uncomment this if you want to save your recorded video to the user's camera roll
      // const { status } = await MediaLibrary.requestPermissionsAsync();

      // if (status !== "granted") {
      //   throw new Error("permission not granted");
      // }

      // const asset = await MediaLibrary.createAssetAsync(recordedVideo.uri)
      // let album = await MediaLibrary.getAlbumAsync("MyAppDrafts");

      // if (album === null) {
      //   album = await MediaLibrary.createAlbumAsync(
      //     "MyAppDrafts",
      //     asset,
      //     false
      //   );
      // } else {
      //   await MediaLibrary.addAssetsToAlbumAsync([asset], album, false);
      // }

      // Auto analyze after recording is done
      await analyzeSignLanguageVideo(recordedVideo.uri);
      
    } catch (error) {
      console.error("[ERROR] Failed to start video recording:", error);
      setIsVideoRecording(false);
    }
  };

  // Stop video recording
  const stopVideoRecording = async () => {
    try {
      console.log("[DEBUG] Stopping video recording...");
      
      if (!cameraRef.current) {
        console.error("[ERROR] Camera reference is null");
        return;
      }
      
      // Use the stopRecording method to stop the video recording
      cameraRef.current.stopRecording();
      setIsVideoRecording(false);
      
      // Note: Analysis will happen when the recording promise resolves in startVideoRecording
      
    } catch (error) {
      console.error("[ERROR] Failed to stop video recording:", error);
      setIsVideoRecording(false);
    }
  };

  // Function to analyze actual video
  const analyzeSignLanguageVideo = async (uri: string) => {
    try {
      console.log("[DEBUG] Preparing video for upload...");
      let formData = new FormData();
      
      // In React Native, we need to append the file differently
      // No need to fetch and create a blob - directly use the uri
      formData.append("file", {
        uri: uri,
        name: "sign_language.mp4",
        type: "video/mp4"
      } as any);
      
      console.log("[DEBUG] Sending video to server for sign language recognition...");
      console.log('[DEBUG] Sending request to: ', HOSTNAME + "recognize-sign-from-video/");
      
      let serverResponse = await fetch(
        HOSTNAME + "recognize-sign-from-video/",
        {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
        }
      );
      
      console.log(`[DEBUG] Server response status: ${serverResponse.status}`);
      
      if (!serverResponse.ok) {
        const errorText = await serverResponse.text();
        console.error("[ERROR] Server response:", errorText);
        throw new Error(errorText);
      }
      
      let data = await serverResponse.json();
      console.log(`[DEBUG] Received sign language response: ${data.recognized_word}`);
      
      // Set the recognized word and speak it out
      if (data.recognized_word) {
        setRecognizedWord(data.recognized_word);
        Speech.speak(data.recognized_word);
      }
    } catch (error) {
      console.error("[ERROR] Error analyzing sign language video:", error);
      // For demonstration: simulate a recognized word if server is unavailable
    }
  };

  const startAudioRecording = async () => {
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

      setAudioRecording(recording);
      setIsAudioRecording(true);
    } catch (err) {
      console.error("Failed to start audio recording:", err);
    }
  };

  const stopAudioRecording = async () => {
    try {
      if (!audioRecording) return;

      await audioRecording.stopAndUnloadAsync();
      setIsAudioRecording(false);
      const uri = audioRecording.getURI();
      setAudioRecording(null);

      if (uri) {
        // Pass the recorded URI to the SpeechToText component
        setTranscriptionUri(uri);
      }
    } catch (err) {
      console.error("Failed to stop audio recording:", err);
    }
  };

  // Handle video recording when user presses play/stop
  const handleVideoRecordingToggle = async () => {
    if (!isVideoRecording) {
      await startVideoRecording();
    } else {
      await stopVideoRecording();
    }
  };

  // Handle audio recording when user presses record/stop
  const handleAudioRecordingToggle = async () => {
    if (!isAudioRecording) {
      await startAudioRecording();
    } else {
      await stopAudioRecording();
    }
  };

  // Clear the sentence
  const clearSentence = () => {
    setSentence([]);
  };
  
  // Remove the last word from the sentence
  const undoLastWord = () => {
    if (sentence.length > 0) {
      setSentence(prev => prev.slice(0, -1));
    }
  };

  return (
    <View style={styles.container}>
      <CameraView 
        style={styles.camera} 
        facing={facing} 
        ref={cameraRef}
        mode="video"
        videoQuality="720p"
      >
        
        {isVideoRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>Recording...</Text>
          </View>
        )}
        
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <AntDesign name="retweet" size={44} color="white" />
          </TouchableOpacity>
          
          {!isVideoRecording ? (
            <TouchableOpacity
              style={styles.button}
              onPress={handleVideoRecordingToggle}
            >
              <AntDesign name="playcircleo" size={44} color="white" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={styles.button}
              onPress={handleVideoRecordingToggle}
            >
              <AntDesign name="pausecircleo" size={44} color="white" />
            </TouchableOpacity>
          )}
          
          {/* Utility buttons moved to the top row */}
          <View style={styles.utilityButtonsGroup}>
            <TouchableOpacity style={styles.utilityButton} onPress={undoLastWord}>
              <Text style={styles.utilityButtonText}>Undo</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.utilityButton} onPress={clearSentence}>
              <Text style={styles.utilityButtonText}>Clear</Text>
            </TouchableOpacity>
          </View>
          
          {!isAudioRecording ? (
            <TouchableOpacity style={styles.button} onPress={handleAudioRecordingToggle}>
              <Text style={styles.buttonText}>Record{"\n"}Audio</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.button} onPress={handleAudioRecordingToggle}>
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

      {/* Display the recognized text from audio */}
      {recognizedText ? (
        <View style={styles.textContainer}>
          <Text style={styles.recognizedText}>
            Recognized: {recognizedText}
          </Text>
        </View>
      ) : null}
      
      {/* Display the accumulated sentence from sign language */}
      {sentence.length > 0 && (
        <View style={styles.sentenceContainer}>
          <Text style={styles.sentenceText}>
            {sentence.join(" ")}
          </Text>
        </View>
      )}
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
  recordingIndicator: {
    position: "absolute",
    top: 20,
    right: 20,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(200, 0, 0, 0.5)",
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 15,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "red",
    marginRight: 8,
  },
  recordingText: {
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
    marginHorizontal: 5,
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
  // Container for utility buttons (now a column in the row)
  utilityButtonsGroup: {
    flexDirection: "column",
    justifyContent: "space-between",
    height: 80,
    marginHorizontal: 5,
  },
  // Style for utility buttons
  utilityButton: {
    width: 70,
    height: 35,
    backgroundColor: "rgba(80, 80, 80, 0.8)",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "white",
    alignItems: "center",
    justifyContent: "center",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1,
    elevation: 2,
  },
  utilityButtonText: {
    fontSize: 14,
    fontWeight: "bold",
    color: "white",
  },
  buttonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
  textContainer: {
    position: "absolute",
    bottom: 120,
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
  sentenceContainer: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    padding: 15,
  },
  sentenceText: {
    color: "white",
    textAlign: "center",
    fontSize: 20,
    fontWeight: "bold",
  },
  text: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
});