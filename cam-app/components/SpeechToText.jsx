import { useEffect } from "react";

export function SpeechToText({ audioUri, onTranscriptionComplete }) {
  useEffect(() => {
    async function transcribe() {
      if (audioUri) {
        try {
          const audioResponse = await fetch(audioUri);
          const audioData = await audioResponse.arrayBuffer();

          const blob = new Blob([audioData], { type: "audio/wav" });

          const formData = new FormData();
          formData.append("audio", blob, "recording.wav");

          const response = await fetch("http://127.0.0.1:8000/process_audio/", {
            method: "POST",
            body: formData,
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
