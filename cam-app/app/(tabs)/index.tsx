import React, { useEffect, useRef } from "react";
import {
  Animated,
  StyleSheet,
  View,
  Text,
  StatusBar,
  TouchableOpacity,
  Dimensions,
  ScrollView,
} from "react-native";
import { useRouter } from "expo-router";

export default function HomeScreen() {
  const fadeImage = useRef(new Animated.Value(0)).current;
  const fadeTitle = useRef(new Animated.Value(0)).current;
  const position = useRef(new Animated.ValueXY({ x: 0, y: 0 })).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const fadeSection = useRef(new Animated.Value(0)).current;

  const { width: wWidth, height: wHeight } = Dimensions.get("window");
  const router = useRouter();

  useEffect(() => {
    const xOffset = -(wWidth / 2) + 85;
    const yOffset = -(wHeight / 2) + 120;

    Animated.sequence([
      // 1) fade in logo
      Animated.timing(fadeImage, {
        toValue: 1,
        duration: 1400,
        useNativeDriver: true,
      }),
      // 2) fade in title
      Animated.timing(fadeTitle, {
        toValue: 1,
        duration: 1200,
        useNativeDriver: true,
      }),
      // 3) pause
      Animated.delay(500),
      // 4) move + shrink hero
      Animated.parallel([
        Animated.timing(position, {
          toValue: { x: xOffset, y: yOffset },
          duration: 1500,
          useNativeDriver: true,
        }),
        Animated.timing(scaleAnim, {
          toValue: 0.6,
          duration: 1500,
          useNativeDriver: true,
        }),
      ]),
      // 5) fade in section
      Animated.timing(fadeSection, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();
  }, [fadeImage, fadeTitle, position, scaleAnim, fadeSection, wWidth, wHeight]);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />

      <ScrollView
        contentContainerStyle={styles.contentContainer}
        showsVerticalScrollIndicator={false}
        bounces={false}
      >
        <Animated.View
          style={[
            styles.hero,
            {
              transform: [
                ...position.getTranslateTransform(),
                { scale: scaleAnim },
              ],
            },
          ]}
        >
          <Animated.Image
            source={require("@/assets/images/horus.png")}
            style={[styles.logo, { opacity: fadeImage }]}
            resizeMode="contain"
          />
          <Animated.Text style={[styles.title, { opacity: fadeTitle }]}>
            HORUS
          </Animated.Text>
        </Animated.View>

        <Animated.View style={[styles.section, { opacity: fadeSection }]}>
          <Text style={styles.headline}>
            Welcome to an immersive ASL experience
          </Text>
          <Text style={styles.description}>
            We use cutting-edge technology and machine learning to detect,
            translate, and teach American Sign Language.
          </Text>

          <Text style={styles.didYouKnow}>Did you know?</Text>
          <View style={styles.facts}>
            <Text style={styles.factItem}>
              <Text style={styles.factNumber}>11 million+</Text> people are deaf
              in the United States
            </Text>
            <Text style={styles.factItem}>
              <Text style={styles.factNumber}>&lt; 1%</Text> of non-deaf people
              know ASL
            </Text>
            <Text style={styles.factItem}>
              <Text style={styles.factNumber}>Zero</Text> free services exist
              for ASL to text & speech
            </Text>
          </View>

          <TouchableOpacity
            style={styles.button}
            onPress={() => router.push("/camera")}
          >
            <Text style={styles.buttonText}>Unlock Free ASL Translation</Text>
          </TouchableOpacity>

          <Text style={styles.footer}>
            Join a vibrant community of thousands, bridging communication gaps
            one sign at a time.
          </Text>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0A1128",
  },
  contentContainer: {
    padding: 20,
  },
  hero: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 20,
  },
  logo: {
    width: 200,
    height: 200,
  },
  title: {
    color: "#FFFFFF",
    fontSize: 28,
    fontWeight: "bold",
    letterSpacing: 2,
    marginTop: 8,
    fontFamily: "Pharaoh",
  },
  section: {
    alignItems: "center",
    paddingHorizontal: 20,
    marginTop: -250,
  },
  headline: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "600",
    marginBottom: 10,
    textAlign: "center",
  },
  description: {
    color: "rgba(203, 213, 225, 0.9)",
    fontSize: 15,
    textAlign: "center",
    marginBottom: 25,
    width: "60%",
    fontFamily: "Pharaoh",
  },
  didYouKnow: {
    color: "#F8FAFC",
    fontSize: 18,
    marginBottom: 8,
    fontWeight: "bold",
    fontFamily: "Georgia",
  },
  facts: {
    alignItems: "center",
    marginBottom: 20,
    width: "100%",
  },
  factItem: {
    color: "#F8FAFC",
    fontSize: 15,
    marginBottom: 6,
    textAlign: "center",
  },
  factNumber: {
    color: "#38BDF8",
    fontSize: 18,
    fontWeight: "bold",
  },
  button: {
    backgroundColor: "#38BDF8",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 30,
  },
  buttonText: {
    color: "#0A1128",
    fontSize: 15,
    fontWeight: "bold",
  },
  footer: {
    color: "rgba(203, 213, 225, 0.8)",
    fontSize: 15,
    textAlign: "center",
    marginTop: 20,
    fontFamily: "Georgia",
  },
});
