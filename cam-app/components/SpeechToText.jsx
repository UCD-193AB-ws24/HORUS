import React, { useState } from "react";
import { View, Text, Button, Platform } from "react-native";
import { StaticTextToSpeech } from "@/components/StaticTextToSpeech";
import * as Speech from "expo-speech";
import { Audio } from "expo-av";

// Change to your local server's address and port
const LOCAL_FLASK_SERVER = "http://127.0.0.1:8000";

export function SpeechToText() {
  // Recording state
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // STT/Whisper state
  const [recognizedText, setRecognizedText] = useState("");
  const [sttError, setSttError] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const [buttonTitle, setButtonTitle] = useState("Start Recording");

  // --- Start Recording ---
  const startRecording = async () => {
    try {
      setSttError(false);
      setErrorMsg("");
      setRecognizedText("");

      // Request permission to record
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        setSttError(true);
        setErrorMsg("Microphone permission not granted");
        return;
      }

      // Configure audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      // Start recording
      const { recording } = await Audio.Recording.createAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      setRecording(recording);
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setSttError(true);
      setErrorMsg("Error starting recording: " + err?.message);
    }
  };

  // --- Stop Recording & Transcribe (Local) ---
  const stopRecordingLocal = async () => {
    try {
      if (!recording) return;

      // Stop & finalize recording
      await recording.stopAndUnloadAsync();
      setIsRecording(false);

      // Retrieve local URI
      const uri = recording.getURI();  // e.g. "file:///data/user/0/.../Audio/..."
      setRecording(null);

      if (uri) {
        await transcribeAudioLocally(uri);
      }
    } catch (err) {
      console.error("Failed to stop recording (local):", err);
      setSttError(true);
      setErrorMsg("Error stopping recording: " + err?.message);
    }
  };

  // --- Upload to Local Flask Whisper Route ---
  const transcribeAudioLocally = async (audioUri) => {
    try {
      // 1) Fetch the recorded file as an ArrayBuffer (on web)
      const audioResponse = await fetch(audioUri);
      const audioData = await audioResponse.arrayBuffer();

      // 2) Turn that ArrayBuffer into a Blob
      //    - For iOS/Android, you might directly append the { uri, type, name } object
      //      But on web, we must create a Blob for the file.
      const blob = new Blob([audioData], { type: "audio/wav" });

      // 3) Create FormData and append the Blob
      const formData = new FormData();
      // The key name here must match FastAPI: if your endpoint uses `audio: UploadFile`
      // then the key is "audio". If it's `file: UploadFile`, use "file" instead.
      formData.append("audio", blob, "recording.wav");

      // 4) POST to your FastAPI server
      const response = await fetch(`${LOCAL_FLASK_SERVER}/process_audio/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to transcribe audio locally: ${errorText}`);
      }

      const data = await response.json();
      console.log("Local server response:", data);

      // 5) Update state based on server response
      if (data.recognized_text) {
        setRecognizedText(data.recognized_text);
      } else if (data.error) {
        setSttError(true);
        setErrorMsg(data.error);
      } else {
        setSttError(true);
        setErrorMsg("No recognized_text returned from local server");
      }
    } catch (err) {
      console.error("Failed to transcribe audio (Local):", err);
      setSttError(true);
      setErrorMsg(err.message);
    }
  };

  return (
    <View style={{ padding: 16 }}>

      {!isRecording && (
        <Button title={buttonTitle} onPress={startRecording} />
      )}

      {isRecording && (
        <>
          <Button
            title="Stop & Transcribe (Local)"
            onPress={stopRecordingLocal}
          />
        </>
      )}

      <Text style={{ marginTop: 16, backgroundColor: "#ffffff", padding: 8 }}>
        Recognized Text: {recognizedText}
      </Text>
    </View>
  );
}
