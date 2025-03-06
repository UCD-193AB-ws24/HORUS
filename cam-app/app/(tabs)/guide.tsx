import { StyleSheet, Image, Platform } from "react-native";

import ParallaxScrollView from "@/components/ParallaxScrollView";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";

export default function TabTwoScreen() {
  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: "#F5F5F5", dark: "#1D3D47" }}
      headerImage={
        <Image
          source={require("@/assets/images/z.png")}
          style={styles.landingLogo}
          resizeMode="contain"
        />
      }
    >
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">How to Sign</ThemedText>
      </ThemedView>
      <Image
        source={require("@/assets/images/asl_alphabet.svg")}
        style={styles.aslAlphabet}
        resizeMode="contain"
      />
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: "#808080",
    bottom: -90,
    left: -35,
    position: "absolute",
  },
  titleContainer: {
    flexDirection: "row",
    gap: 8,
  },
  landingLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: "static",
  },
  aslAlphabet: {
    height: 300,
    width: "100%",
    marginTop: 20,
    backgroundColor: "transparent",
  },
});
