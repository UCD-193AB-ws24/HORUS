import { useEffect } from "react";

export function SpeechToText({ audioUri, onTranscriptionComplete }) {
  useEffect(() => {
    async function transcribe() {
      if (audioUri) {
        try {
          const formData = new FormData();
          formData.append("file", {
            uri: audioUri,
            name: "recording.wav",
            type: "audio/wav"
          });

          const response = await fetch("https://0f4e-2600-1010-b33b-f128-95dc-1556-e9d7-91c6.ngrok-free.app/process_audio/", {
            method: "POST",
            body: formData,
            headers: {
              Accept: "application/json",
            },
          });

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Transcription failed: ${errorText}`);
          }

          const data = await response.json();
          if (data.recognized_text) {
            onTranscriptionComplete(data.recognized_text);
          } else {
            onTranscriptionComplete("");
          }
        } catch (error) {
          console.error("Transcription error:", error);
          onTranscriptionComplete("");
        }
      }
    }
    transcribe();
    console.log("finished transcribing");
  }, [audioUri, onTranscriptionComplete]);

  return null;
}
