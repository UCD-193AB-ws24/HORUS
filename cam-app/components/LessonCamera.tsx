import { useEffect, useRef, useState } from "react";
import { View } from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";

let HOSTNAME =
  "https://0e7e-2600-1010-b13f-54d4-c866-d922-d45b-8039.ngrok-free.app/";

type Props = {
  onDetect(letter: string): void;
  intervalMs?: number;
};

export default function LessonCamera({ onDetect, intervalMs = 300 }: Props) {
  const [facing] = useState<CameraType>("front");
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

        // const blob = await (await fetch(photo.uri)).blob();

        let formData = new FormData();
        // form.append("file", blob, "frame.jpg");
        formData.append("file", {
          uri: photo.uri,
          name: "photo.jpg",
          type: "image/jpg",
        } as any);

        const res = await fetch(HOSTNAME + "recognize-gesture/", {
          method: "POST",
          body: formData,
          headers: { Accept: "application/json" },
        });

        if (!res.ok) {
          const errorText = await res.text();
          console.error(errorText);
        }

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

  return (
    <CameraView
      ref={cameraRef}
      style={{ flex: 1 }}
      facing={facing}
      animateShutter={false}
    />
  );
}
