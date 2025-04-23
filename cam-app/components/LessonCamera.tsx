import { useEffect, useRef, useState } from "react";
import { View } from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";

type Props = {
  onDetect(letter: string): void;
  intervalMs?: number;
};

export default function LessonCamera({ onDetect, intervalMs = 300 }: Props) {
  const [facing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();

  const cameraRef = useRef<CameraView | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (permission && !permission.granted) requestPermission();
  }, [permission, requestPermission]);

  useEffect(() => {
    if (!permission?.granted) return;

    const tick = async () => {
      if (!cameraRef.current) return;

      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.5,
          base64: true,
          exif: false,
        });

        if (!photo?.uri) return;

        const blob = await (await fetch(photo.uri)).blob();

        const form = new FormData();
        form.append("file", blob, "frame.jpg");

        const res = await fetch("http://127.0.0.1:8000/recognize-gesture/", {
          method: "POST",
          body: form,
          headers: { Accept: "application/json" },
        });

        if (!res.ok) return;

        const { gesture } = (await res.json()) as { gesture?: string };
        if (gesture && gesture !== "None") onDetect(gesture.toUpperCase());
      } catch {
        // ignore
      }
    };

    intervalRef.current = setInterval(tick, intervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [permission?.granted, intervalMs, onDetect]);

  if (!permission?.granted) return <View style={{ flex: 1 }} />;

  return <CameraView ref={cameraRef} style={{ flex: 1 }} facing={facing} />;
}
