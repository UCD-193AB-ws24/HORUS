import { Image, StyleSheet, Platform, Button } from "react-native";

import {SpeechToText} from "@/components/SpeechToText";
import {StaticTextToSpeech} from "@/components/StaticTextToSpeech";
import { HelloWave } from "@/components/HelloWave";
import ParallaxScrollView from "@/components/ParallaxScrollView";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { useEffect, useState } from "react";


export default function HomeScreen() {
  const [input, setInput] = useState<string>("");

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: "#A1CEDC", dark: "#1D3D47" }}
      headerImage={
        <Image
          source={require("@/assets/images/partial-react-logo.png")}
          style={styles.reactLogo}
          resizeMode="cover"
        />
      }
    >
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Welcome!</ThemedText>
        <HelloWave />
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">ASL Detection & Translation App</ThemedText>
        <ThemedText>
          Welcome to our American Sign Langauge detection and translation app.
          To get started, head over to the camera tab and enable permissions!
        </ThemedText>
        <StaticTextToSpeech input={input}/>
        <SpeechToText/>
      </ThemedView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: "absolute",
  },
  fullScreenImage: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
});
