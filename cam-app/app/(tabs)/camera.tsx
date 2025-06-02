import { SetStateAction, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  TextInput,
  Dimensions,
} from "react-native";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { AntDesign, Entypo, Ionicons } from "@expo/vector-icons";
import * as Speech from "expo-speech";
import { Colors } from "@/constants/Colors";
import { Audio } from "expo-av";
import { SpeechToText } from "@/components/SpeechToText";
import SigningTimingBar from "@/components/SigningTimingBar";
import { useIsFocused } from "@react-navigation/native";

let HOSTNAME = "https://capstoneserver193.duckdns.org/";
import Checkbox from "expo-checkbox";
import { useAuth } from "@/lib/AuthContext";
import { supabase } from "@/lib/supabaseClient";
import { decode, encode } from "base64-arraybuffer";

export default function CameraComponent() {
  const isFocused = useIsFocused();
  const [facing, setFacing] = useState<CameraType>("front");
  const [permission, requestPermission] = useCameraPermissions();
  const [recognizedWord, setRecognizedWord] = useState<string | null>(null);
  const [isVideoRecording, setIsVideoRecording] = useState(false);
  const [isAudioRecording, setIsAudioRecording] = useState(false);
  const [audioRecording, setAudioRecording] = useState<Audio.Recording | null>(
    null
  );
  const [recognizedText, setRecognizedText] = useState("");
  const [transcriptionUri, setTranscriptionUri] = useState<string | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [recordingPhase, setRecordingPhase] = useState<
    "idle" | "prepare" | "record" | "complete"
  >("idle");
  const [needHelp, setNeedHelp] = useState(false);
  const [signSigned, onChangeSignSigned] = useState("");
  const [signTranslated, onChangeSignTranslated] = useState("");
  const [isChecked, setChecked] = useState(false);
  const { user } = useAuth();
  
  const cameraRef = useRef<CameraView | null>(null);
  const recordingTimer = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    console.log("[DEBUG] Checking camera permissions...");
  }, []);

  // Add recognized words to our sentence
  useEffect(() => {
    if (recognizedWord && recognizedWord !== "None") {
      // Only add the word if it's new (not already the last word in the sentence)
      setSentence((prev) => {
        // Check if this word is already the last word in the array (to avoid duplicates)
        if (prev.length > 0 && prev[prev.length - 1] === recognizedWord) {
          return prev;
        }
        return [...prev, recognizedWord];
      });
    }
  }, [recognizedWord]);

  if (!isFocused) {
    return <View style={styles.container} />;
  }

  if (!permission) {
    console.log("[DEBUG] Camera permissions are still loading...");
    return <View />;
  }
  if (!permission.granted) {
    console.log("[DEBUG] Camera permissions not granted");
    return (
      <View style={styles.container}>
        <Text style={{ fontSize: 22, textAlign: "center" }}>
          We need your permission to show the camera
        </Text>
        <Text>{"\n"}</Text>
        <View style={styles.permissionsButtonContainer}>
          <TouchableOpacity
            onPress={requestPermission}
            style={styles.permissionsButton}
          >
            <Text style={styles.text}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  function toggleCameraFacing() {
    console.log("[DEBUG] Toggling camera facing...");
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  // Handle recording phase changes
  const handleRecordingPhaseChange = (
    phase: "idle" | "prepare" | "record" | "complete"
  ) => {
    // Only update if the phase is actually changing
    if (recordingPhase !== phase) {
      console.log(`[DEBUG] Recording phase changed to: ${phase}`);
      setRecordingPhase(phase);

      if (phase === "complete") {
        // Automatically stop recording when timing bar completes
        stopVideoRecording();
      }
    }
  };

  // Actually start video recording
  const startVideoRecording = async () => {
    try {
      console.log("[DEBUG] Starting actual video recording...");

      if (!cameraRef.current) {
        console.error("[ERROR] Camera reference is null");
        return;
      }

      setIsVideoRecording(true);
      setRecordingPhase("prepare"); // Initialize phase
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Using the recordAsync method with updated options
      const videoRecordPromise = cameraRef.current.recordAsync({
        maxDuration: 1, // Maximum duration in seconds
        codec: "avc1",
      });

      // Set up the promise resolution
      const recordedVideo = await videoRecordPromise;
      console.log("[DEBUG] Video recording completed:", recordedVideo.uri);
      setVideoUri(recordedVideo.uri);

      // Always analyze after successful recording completion
      console.log("[DEBUG] Analyzing sign language video...");
      await analyzeSignLanguageVideo(recordedVideo.uri);
      
    } catch (error) {
      console.error("[ERROR] Failed to start video recording:", error);
    } finally {
      setIsVideoRecording(false);
      setRecordingPhase("idle");
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

      // Note: Analysis will happen when the recording promise resolves in startVideoRecording
    } catch (error) {
      console.error("[ERROR] Failed to stop video recording:", error);
      setIsVideoRecording(false);
      setRecordingPhase("idle");
    }
  };

  const uploadVideoToBucket = async (uri: string, sign: string) => {
    const d = new Date();
    let time = d.getTime();
    let file_name = user.id + "/" + sign + ".mp4";
    const response = await fetch(uri);
    const blob = await response.blob();
    let fileReader = new FileReader();
    let array;
    fileReader.readAsArrayBuffer(blob);
    fileReader.onload = async function () {
      array = this.result;
      let buf = encode(array);
      const { data, error } = await supabase.storage
        .from("video-files")
        .update(file_name, decode(buf), {
          contentType: "video/mp4",
        });
      if (error) {
        console.error("Unable to upload file", error);
      }
    };
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
        type: "video/mp4",
      } as any);

      console.log(
        "[DEBUG] Sending video to server for sign language recognition..."
      );
      console.log(
        "[DEBUG] Sending request to: ",
        HOSTNAME + "recognize-sign-from-video/"
      );

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
      console.log(
        `[DEBUG] Received sign language response: ${data.recognized_word}`
      );

      // Set the recognized word and speak it out
      if (data.recognized_word) {
        setRecognizedWord(data.recognized_word);
        Speech.speak(data.recognized_word);
      }
      await uploadVideoToBucket(uri, data.recognized_word);
    } catch (error) {
      console.error("[ERROR] Error analyzing sign language video:", error);
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
    setRecognizedWord(null);
    setRecognizedText("");
  };

  // Remove the last word from the sentence
  const undoLastWord = () => {
    if (sentence.length > 0) {
      setSentence((prev) => prev.slice(0, -1));
    }
  };

  const moveVideo = async (fileName: string) => {
    let t = new Date().toLocaleTimeString("en-US", {
      hour12: false,
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
    });
    const { data, error } = await supabase.storage
      .from("video-files")
      .move(
        user.id + "/" + fileName,
        "incorrect-translations/" + signSigned.toLowerCase() + "-" + t + ".mp4"
      );
    if (error) {
      console.error("Unable to move file ", error);
    }
  };

  const moveErrorVideo = async () => {
    if (isChecked) {
      const { data, error } = await supabase.storage
        .from("video-files")
        .list(String(user.id));
      if (!error) {
        data.forEach((element) => {
          let fileName = element.name;
          let nameArray = fileName.split(".");
          if (nameArray[0] == signTranslated.toLowerCase()) {
            moveVideo(fileName);
          }
        });
      }
    }
    onChangeSignSigned("");
    onChangeSignTranslated("");
    setChecked(false);
    setNeedHelp(false);
  };

  const exitHelpPage = () => {
    onChangeSignSigned("");
    onChangeSignTranslated("");
    setChecked(false);
    setNeedHelp(false);
  };

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={facing}
        ref={cameraRef}
        mode="video"
        videoQuality="480p"
      >
        {/* Signing Timing Bar */}
        <SigningTimingBar
          isRecording={isVideoRecording}
          totalDuration={2000} // 3 seconds total
          preparationTime={1000} // 1 second preparation time
          onRecordingPhaseChange={handleRecordingPhaseChange}
        />

        {/*Recording indicator*/}
        {isVideoRecording && (
          <View style={styles.recordingIndicatorLandscape}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>Recording...</Text>
          </View>
        )}

        
        <View style={styles.recordingButtonContainerLandscape}>
          {/*Undo and clear buttons*/}
          {/* Utility buttons moved to the top row */}
          <View style={styles.utilityButtonsGroup}>
            <TouchableOpacity
              style={styles.utilityButton}
              onPress={undoLastWord}
            >
              <AntDesign name="back" size={25} color="white" />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.utilityButton}
              onPress={clearSentence}
            >
              <AntDesign name="delete" size={25} color="white" />
            </TouchableOpacity>
          </View>
          {/*Start recording and stop recording buttons*/}
          {!isVideoRecording ? (
            <TouchableOpacity
              style={styles.recordButtonLandscape}
              onPress={handleVideoRecordingToggle}
            >
              <AntDesign name="playcircleo" size={44} color="white" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={styles.recordButtonLandscape}
              onPress={handleVideoRecordingToggle}
            >
              {/* <AntDesign name="pausecircleo" size={44} color="white" /> */}
              <Entypo name="controller-stop" size={44} color="red" />
            </TouchableOpacity>
          )}
          {/*Flip camera button*/}
          {/*
          <TouchableOpacity
            style={styles.flipCameraButton}
            onPress={toggleCameraFacing}
          >
            <AntDesign name="retweet" size={40} color="white" />
          </TouchableOpacity>
          */}
        </View>

        {/*Record audio buttons*/}
        <View style={styles.audioButton}>
          {!isAudioRecording ? (
            <TouchableOpacity
              style={styles.buttonLandscape}
              onPress={handleAudioRecordingToggle}
            >
              <Ionicons name="mic" size={40} color="white" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={styles.buttonLandscape}
              onPress={handleAudioRecordingToggle}
            >
              <Ionicons name="mic" size={40} color="red" />
            </TouchableOpacity>
          )}
        </View>
    
       {/*Report and error button*/}
        <View style={styles.helpButton}>
          <TouchableOpacity
            style={styles.buttonLandscape}
            onPress={() => setNeedHelp(true)}
          >
            <AntDesign name="customerservice" size={40  } color="white" />
          </TouchableOpacity>
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
            Recognized Speech: {recognizedText}
          </Text>
        </View>
      ) : null}

      {/* Display the accumulated sentence from sign language */}
      {sentence.length > 0 && (
        <View style={styles.sentenceContainer}>
          <Text style={styles.sentenceText}>{sentence.join(" ")}</Text>
        </View>
      )}

      {/*Report and error pop up window*/}
      <Modal
        animationType="slide"
        transparent={true}
        visible={needHelp}
        supportedOrientations={["landscape"]}
      >
        <View style={styles.centeredView}>
          <View style={styles.modalView}>
            <Text style={styles.modalTextLandscape}>
              If you would like to report a wrong translation please fill out
              the boxes below:
            </Text>
            <Text style={styles.modalTextLandscape}>
              {"\n"}What sign did you sign?
            </Text>
            <TextInput
              style={styles.input}
              onChangeText={onChangeSignSigned}
              value={signSigned}
            />
            <Text style={styles.modalTextLandscape}>
              What sign was translated?
            </Text>
            <TextInput
              style={styles.input}
              onChangeText={onChangeSignTranslated}
              value={signTranslated}
            />
            <Text style={styles.modalTextLandscape}>
              {"\n"}Would you like to include the video of you signing?
            </Text>
            <Checkbox
              style={styles.checkbox}
              value={isChecked}
              onValueChange={setChecked}
              color={isChecked ? "#4630EB" : undefined}
            />
          </View>
          <View style={styles.modalButton}>
            <TouchableOpacity style={styles.button} onPress={exitHelpPage}>
              <AntDesign name="close" size={44} color="white" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={moveErrorVideo}>
              <AntDesign name="check" size={44} color="white" />
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
 
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
  recordingIndicatorLandscape: {
    position: "absolute",
    top: 20,
    left: 100,
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
  recordingButtonContainerLandscape: {
    display: "flex",
    flexDirection: "column",
    backgroundColor: "transparent",
    justifyContent: "space-evenly",
    alignItems: "flex-end",
    position: "fixed",
    gap: 65,
    marginTop: 20,
    marginLeft: 10,
  },
  audioButton: {
    width: 75,
    height: 80,
    position: "absolute",
    bottom: 0,
    left: 5,
  },
  helpButton: {
    width: 75,
    height: 80,
    position: "absolute",
    top: 20,
    left: 5,
  },
  button: {
    width: 75,
    height: 80,
    marginHorizontal: 5,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonLandscape: {
    width: 60,
    height: 60,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    left: 10,
  },
  flipCameraButton: {
    width: 60,
    height: 60,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    top: 10,
    right: 15,
  },
  recordButtonLandscape: {
    width: 60,
    height: 120,
    right: 15,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  permissionsButton: {
    width: 120,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  // Container for utility buttons (now a column in the row)
  utilityButtonsGroup: {
    flexDirection: "column",
    justifyContent: "space-between",
    right: 5,
    height: 70,
    marginHorizontal: 5,
  },
  // Style for utility buttons
  utilityButton: {
    width: 70,
    height: 30,
    borderRadius: 40,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
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
    bottom: 80,
    left: 100,
    // top: 10,
    maxWidth: "35%",
    zIndex: 10,
    padding: 5,
  },
  recognizedText: {
    color: "white",
    fontSize: 20,
    flexWrap: "wrap",
    backgroundColor: "rgba(0, 0, 0, 0.3)",
  },
  sentenceContainer: {
    position: "absolute",
    left: 469,
    bottom: 80,
    maxWidth: "35%",
    zIndex: 10,
    padding: 5,
    alignItems: "flex-end",
  },
  sentenceText: {
    color: "white",
    textAlign: "right",
    fontSize: 20,
    flexWrap: "wrap",
    backgroundColor: "rgba(0, 0, 0, 0.3)",
  },
  text: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
  },
  centeredView: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  modalView: {
    margin: 20,
    backgroundColor: "white",
    borderRadius: 20,
    padding: 35,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  modalTextLandscape: {
    fontSize: 12,
    fontWeight: "bold",
    color: "black",
    textAlign: "center",
  },
  input: {
    height: 40,
    width: 75,
    margin: 12,
    borderWidth: 1,
    padding: 10,
  },
  modalButton: {
    flexDirection: "row",
    backgroundColor: "transparent",
    justifyContent: "center",
    alignItems: "flex-end",
    marginBottom: 20,
  },
  checkbox: {
    height: 20,
    width: 20,
    margin: 8,
  },
  permissionsButtonContainer: {
    flexDirection: "row",
    backgroundColor: "transparent",
    justifyContent: "center",
    alignItems: "flex-end",
    marginBottom: 20,
  },
});
