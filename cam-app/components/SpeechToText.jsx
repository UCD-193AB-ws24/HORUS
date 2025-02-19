import React, { useState, useEffect } from "react";
import { View, Text, Button } from "react-native";
import Voice from "@react-native-voice/voice";

export function SpeechToText() {
  const [isListening, setIsListening] = useState<boolean>(false);
  const [recognizedText, setRecognizedText] = useState<string>("");
  const [sttError, setSttError] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  useEffect(() => {
    Voice.onSpeechStart = () => setIsListening(true);
    Voice.onSpeechEnd = () => setIsListening(false);
    Voice.onSpeechError = (error) => {
        console.error("Speech error:", error)
        setErrorMsg(error.error.message);
        setSttError(true);
    };
    Voice.onSpeechResults = (result) => {
      if (result.value?.length) {
        setRecognizedText(result.value[0]);
      }
    };

    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  const startListening = async () => {
    try {
      await Voice.start("en-US");
    } catch (error) {
      console.error("Error starting voice recognition:", error);
    }
  };

  const stopListening = async () => {
    try {
      await Voice.stop();
    } catch (error) {
      console.error("Error stopping voice recognition:", error);
    }
  };

  return (
    <View>
        {sttError ? (<Text>Speech recognition failed with error {errorMsg}</Text>) : (
            <>
                <Text>{isListening ? "Listening..." : "Not listening"}</Text>
                <Button title="Start Listening" onPress={startListening} />
                <Button title="Stop Listening" onPress={stopListening} />
                <Text>Recognized Text: {recognizedText}
                </Text>
            </>
        )}
    </View>
  );
};
