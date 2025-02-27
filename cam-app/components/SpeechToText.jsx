import React, { useState } from "react";
import { View, Text, TextInput, Button, Platform } from "react-native";
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

  // Text to Speak
  const [textToSpeak, setTextToSpeak] = useState("");

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

  // --- Stop Recording & Transcribe with Remote OpenAI Whisper ---
  const stopRecordingRemote = async () => {
    try {
      if (!recording) return;

      // Stop & finalize recording
      await recording.stopAndUnloadAsync();
      setIsRecording(false);

      // Retrieve local URI
      const uri = recording.getURI();
      setRecording(null);

      if (uri) {
        await transcribeAudioWithOpenAI(uri);
      }
    } catch (err) {
      console.error("Failed to stop recording (remote):", err);
      setSttError(true);
      setErrorMsg("Error stopping recording: " + err?.message);
    }
  };

  // --- Stop Recording & Transcribe with Local Flask Server ---
  const stopRecordingLocal = async () => {
    try {
      if (!recording) return;

      // Stop & finalize recording
      await recording.stopAndUnloadAsync();
      setIsRecording(false);

      // Retrieve local URI
      const uri = recording.getURI();
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

  // --- Remote OpenAI Whisper API call ---
  const transcribeAudioWithOpenAI = async (uri) => {
    try {
      const fileUri = uri.replace("file://", "");

      // Build multipart form data
      const formData = new FormData();
      formData.append("model", "whisper-1");
      formData.append("file", {
        uri: Platform.OS === "android" ? uri : fileUri,
        type: "audio/m4a",
        name: "audio.m4a",
      });

      const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "multipart/form-data",
        },
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error("Whisper API error: " + errText);
      }

      const data = await response.json();
      if (data && data.text) {
        setRecognizedText(data.text);
      } else {
        setSttError(true);
        setErrorMsg("No text returned from Whisper");
      }
    } catch (err) {
      console.error("Failed to transcribe audio (OpenAI):", err);
      setSttError(true);
      setErrorMsg(err.message);
    }
  };

  // --- Local Flask Whisper route call ---
  const transcribeAudioLocally = async (audioUri) => {
    try {
      // Using "file:///..." can cause issues on iOS/Android. We'll just send the raw path.
      const formData = new FormData();
      formData.append("audio", {
        uri: audioUri,
        type: "audio/wav", // If your local server expects wav. Adjust if m4a, mp4, etc.
        name: "recording.wav",
      });

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
      {/* Example TTS component */}
      <StaticTextToSpeech />

      {/* STT Section */}
      {sttError && (
        <Text style={{ color: "red", marginTop: 16 }}>Error: {errorMsg}</Text>
      )}

      <Text style={{ marginTop: 16 }}>
        {isRecording ? "Recording..." : "Not recording"}
      </Text>

      {!isRecording && <Button title="Start Recording" onPress={startRecording} />}

      {isRecording && (
        <>
          {/* Another button to stop & use Local Flask */}
          <Button title="Stop & Transcribe (Local)" onPress={stopRecordingLocal} />
        </>
      )}

      <Text style={{ marginTop: 16 }}>Recognized Text: {recognizedText}</Text>
    </View>
  );
}
