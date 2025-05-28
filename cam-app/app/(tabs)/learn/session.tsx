import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  View,
  Text,
  Image,
  Pressable,
  StyleSheet,
  StatusBar,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
  SafeAreaView,
  Animated,
} from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { useIsFocused } from "@react-navigation/native";
import useAlphabetLesson from "@/hooks/useAlphabetLesson";
import LessonCamera from "@/components/LessonCamera";
import { useAuth } from "@/lib/AuthContext";
import { Ionicons } from "@expo/vector-icons";

const signImages: Record<string, number> = {
  a: require("@/assets/signs/a.png"),
  b: require("@/assets/signs/b.png"),
  c: require("@/assets/signs/c.png"),
  d: require("@/assets/signs/d.png"),
  e: require("@/assets/signs/e.png"),
  f: require("@/assets/signs/f.png"),
  g: require("@/assets/signs/g.png"),
  h: require("@/assets/signs/h.png"),
  i: require("@/assets/signs/i.png"),
  j: require("@/assets/signs/j.png"),
  k: require("@/assets/signs/k.png"),
  l: require("@/assets/signs/l.png"),
  m: require("@/assets/signs/m.png"),
  n: require("@/assets/signs/n.png"),
  o: require("@/assets/signs/o.png"),
  p: require("@/assets/signs/p.png"),
  q: require("@/assets/signs/q.png"),
  r: require("@/assets/signs/r.png"),
  s: require("@/assets/signs/s.png"),
  t: require("@/assets/signs/t.png"),
  u: require("@/assets/signs/u.png"),
  v: require("@/assets/signs/v.png"),
  w: require("@/assets/signs/w.png"),
  x: require("@/assets/signs/x.png"),
  y: require("@/assets/signs/y.png"),
  z: require("@/assets/signs/z.png"),
};

const CONGRATULATORY_MESSAGES = [
  "Good job!",
  "That was awesome!",
  "You're amazing!",
  "Perfect!",
  "Keep it up!",
  "You're a natural!",
  "Excellent work!",
];

// Custom Switch Component
function CustomSwitch({
  value,
  onValueChange,
}: {
  value: boolean;
  onValueChange: (value: boolean) => void;
}) {
  const translateX = useRef(new Animated.Value(value ? 22 : 2)).current;

  const toggleSwitch = () => {
    const newValue = !value;
    Animated.spring(translateX, {
      toValue: newValue ? 22 : 2,
      bounciness: 4,
      speed: 12,
      useNativeDriver: true,
    }).start();
    onValueChange(newValue);
  };

  return (
    <Pressable
      style={[
        styles.switchContainer,
        value ? styles.switchActive : styles.switchInactive,
      ]}
      onPress={toggleSwitch}
      accessibilityRole="switch"
      accessibilityState={{ checked: value }}
    >
      <Animated.View
        style={[
          styles.switchThumb,
          value ? styles.switchThumbActive : styles.switchThumbInactive,
          { transform: [{ translateX }] },
        ]}
      >
        {value && (
          <Ionicons
            name="eye"
            size={12}
            color="#fff"
            style={styles.switchIcon}
          />
        )}
        {!value && (
          <Ionicons
            name="eye-off"
            size={12}
            color="#64748B"
            style={styles.switchIcon}
          />
        )}
      </Animated.View>
    </Pressable>
  );
}

export default function Session() {
  const router = useRouter();
  const isFocused = useIsFocused();
  const { startIndex } = useLocalSearchParams<{ startIndex?: string }>();
  const { letters } = useAlphabetLesson();
  const { saveProgress, user } = useAuth();
  const initialIndex = parseInt(startIndex || "0", 10);
  const [index, setIndex] = useState(initialIndex);
  const [score, setScore] = useState(initialIndex * 50);
  const [congratulatoryMessage, setCongratulatoryMessage] = useState("");
  const [showImage, setShowImage] = useState(false);
  const lock = useRef(false);

  const currentLetter = useMemo(() => letters[index], [letters, index]);

  useEffect(() => {
    const saveCurrentProgress = async () => {
      if (user && index > 0) {
        await saveProgress(index);
      }
    };

    return () => {
      saveCurrentProgress();
    };
  }, [index, saveProgress, user]);

  useEffect(() => {
    if (!user || index === 0) return;

    const saveInterval = setInterval(async () => {
      await saveProgress(index);
    }, 10000);

    return () => {
      clearInterval(saveInterval);
    };
  }, [index, saveProgress, user]);

  const handleDetect = useCallback(
    (letter: string) => {
      if (lock.current || letter !== currentLetter) return;
      lock.current = true;

      const nextScore = score + 50;
      const rand = Math.floor(Math.random() * CONGRATULATORY_MESSAGES.length);
      setCongratulatoryMessage(CONGRATULATORY_MESSAGES[rand]);

      if (index === letters.length - 1) {
        if (user) {
          saveProgress(0);
        }

        router.replace({
          pathname: "/(tabs)/learn/result",
          params: { score: String(nextScore) },
        });
        return;
      }

      const nextIndex = index + 1;
      setScore(nextScore);
      setIndex(nextIndex);

      if (user) {
        saveProgress(nextIndex);
      }

      setTimeout(() => {
        lock.current = false;
      }, 500);
    },
    [currentLetter, index, letters.length, router, score, saveProgress, user]
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.backgroundGradient} />

      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <Pressable
              accessibilityRole="button"
              onPress={() => router.back()}
              style={styles.backButton}
            >
              <Ionicons name="arrow-back" size={28} color="#38BDF8" />
            </Pressable>
            <View style={styles.headerCenter}>
              <View style={styles.centeredInstructionContainer}>
                <View style={styles.instructionRow}>
                  <Text style={styles.letterInstruction}>
                    Sign the letter {currentLetter}
                  </Text>
                  <Text style={styles.scoreText}>Score: {score}</Text>
                </View>
                {congratulatoryMessage !== "" && (
                  <Text style={styles.congratsMessage}>
                    {congratulatoryMessage}
                  </Text>
                )}
              </View>
            </View>
            <View style={styles.switchWrapper}>
              <Text style={styles.switchLabel}>Show Letter</Text>
              <CustomSwitch value={showImage} onValueChange={setShowImage} />
            </View>
          </View>

          <View
            style={
              showImage ? styles.sideBySideContainer : styles.centerContainer
            }
          >
            <View
              style={[
                styles.cameraContainer,
                showImage && styles.cameraContainerSmaller,
              ]}
            >
              {isFocused && <LessonCamera onDetect={handleDetect} />}
            </View>

            {showImage && (
              <View style={styles.signImageContainer}>
                <Image
                  source={signImages[currentLetter.toLowerCase()]}
                  style={styles.signImage}
                  resizeMode="contain"
                />
              </View>
            )}
          </View>

          {/* Add extra space at bottom to account for navbar */}
          <View style={styles.bottomSpacer} />
        </ScrollView>
      </TouchableWithoutFeedback>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0A1128",
  },
  scrollContent: {
    flexGrow: 1,
    padding: 16,
  },
  backgroundGradient: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "#0A1128",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 10,
    marginBottom: 16,
    paddingHorizontal: 8,
  },
  headerCenter: {
    flex: 1,
    alignItems: "center",
    marginHorizontal: 8,
  },
  backButton: {
    padding: 8,
    width: 44,
  },
  switchWrapper: {
    flexDirection: "row",
    alignItems: "center",
    width: 180,
    flexShrink: 0,
    flexWrap: "nowrap",
  },
  switchLabel: {
    color: "#E2E8F0",
    marginRight: 12,
    fontSize: 16,
    fontFamily: "Pharaoh",
    flexShrink: 0,
  },
  switchContainer: {
    width: 48,
    height: 26,
    borderRadius: 20,
    justifyContent: "center",
    padding: 2,
  },
  switchInactive: {
    backgroundColor: "rgba(71, 85, 105, 0.7)",
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.5)",
  },
  switchActive: {
    backgroundColor: "rgba(56, 189, 248, 0.25)",
    borderWidth: 1,
    borderColor: "rgba(56, 189, 248, 0.7)",
  },
  switchThumb: {
    width: 22,
    height: 22,
    borderRadius: 11,
    justifyContent: "center",
    alignItems: "center",
  },
  switchThumbInactive: {
    backgroundColor: "#E2E8F0",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 3,
  },
  switchThumbActive: {
    backgroundColor: "#38BDF8",
    shadowColor: "#38BDF8",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 3,
    elevation: 5,
  },
  switchIcon: {
    alignSelf: "center",
  },
  centerContainer: {
    alignItems: "center",
  },
  sideBySideContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  cameraContainer: {
    alignSelf: "center",
    width: "90%",
    height: 300,
    overflow: "hidden",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(56, 189, 248, 0.3)",
  },
  cameraContainerSmaller: {
    width: "70%",
    height: 250,
  },
  signImageContainer: {
    width: "40%",
    alignItems: "center",
    justifyContent: "center",
  },
  signImage: {
    width: 120,
    height: 120,
    borderRadius: 12,
    backgroundColor: "rgba(15, 23, 42, 0.3)",
    borderWidth: 1,
    borderColor: "rgba(56, 189, 248, 0.3)",
    padding: 6,
  },
  centeredInstructionContainer: {
    alignItems: "center",
  },
  instructionRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
  },
  letterInstruction: {
    fontSize: 24,
    fontWeight: "600",
    color: "#F8FAFC",
    marginBottom: 0,
    fontFamily: "Pharaoh",
  },
  congratsMessage: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#4ADE80",
    marginBottom: 4,
  },
  scoreText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#38BDF8",
    marginLeft: 16,
    marginBottom: 0,
  },
  bottomSpacer: {
    height: 80,
  },
});
