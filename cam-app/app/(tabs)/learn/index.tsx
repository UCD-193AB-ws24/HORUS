import {
  View,
  Pressable,
  Text,
  StyleSheet,
  StatusBar,
  Image,
  ScrollView,
} from "react-native";
import { useRouter } from "expo-router";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/AuthContext";
import useAlphabetLesson from "@/hooks/useAlphabetLesson";

export default function AlphabetLessonStart() {
  const router = useRouter();
  const { getProgress, user, saveProgress } = useAuth();
  const { letters } = useAlphabetLesson();
  const [savedIndex, setSavedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProgress = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        const { letterIndex } = await getProgress();
        if (letterIndex > 0 && letterIndex < letters.length) {
          setSavedIndex(letterIndex);
        }
      } catch (error) {
        console.error("Error fetching progress:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchProgress();
  }, [user, getProgress, letters.length]);

  const startNewSession = async () => {
    if (user) {
      await saveProgress(0);
      setSavedIndex(null);
    }

    router.push({
      pathname: "/(tabs)/learn/session",
      params: { startIndex: "0" },
    });
  };

  const continueSession = () => {
    if (savedIndex === null) return;

    router.push({
      pathname: "/(tabs)/learn/session",
      params: { startIndex: savedIndex.toString() },
    });
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" />
        <View style={styles.backgroundGradient} />
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.backgroundGradient} />
      <ScrollView
        style={styles.scrollContainer}
        contentContainerStyle={styles.contentContainerStyle}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.contentContainer}>
          <View style={styles.headerSection}>
            <Image
              source={require("@/assets/images/horus.png")}
              style={styles.logo}
              resizeMode="contain"
            />
            <Text style={styles.appTitle}>HORUS</Text>
          </View>

          <View style={styles.formContainer}>
            <Text style={styles.title}>Learn the ASL Alphabet Today</Text>
            <Text style={styles.subtitle}>
              Interactive lessons tailored to your pace
            </Text>

            <View style={styles.buttonContainer}>
              {savedIndex !== null && (
                <Pressable
                  accessibilityRole="button"
                  onPress={continueSession}
                  style={[styles.button, styles.secondaryButton]}
                >
                  <Text style={styles.buttonText}>
                    Continue from letter {letters[savedIndex]}
                  </Text>
                </Pressable>
              )}

              <Pressable
                accessibilityRole="button"
                onPress={startNewSession}
                style={[
                  styles.button,
                  savedIndex !== null
                    ? styles.secondaryButton
                    : styles.primaryButton,
                ]}
              >
                <Text style={styles.buttonText}>
                  {savedIndex !== null
                    ? "Start from beginning"
                    : "Begin your ASL learning journey"}
                </Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.footerContainer}>
            <Text style={styles.footerText}>
              Practice your American Sign Language skills
            </Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0A1128",
  },
  backgroundGradient: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "#0A1128",
  },
  scrollContainer: {
    flex: 1,
  },
  contentContainerStyle: {
    flexGrow: 1,
  },
  contentContainer: {
    flex: 1,
    justifyContent: "space-between",
    padding: 20,
  },
  headerSection: {
    alignItems: "center",
    justifyContent: "center",
  },
  logo: {
    width: 100,
    height: 100,
  },
  appTitle: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "bold",
    letterSpacing: 4,
    textAlign: "center",
    textTransform: "uppercase",
    fontFamily: "Pharaoh",
  },
  formContainer: {
    width: "100%",
    alignItems: "center",
    padding: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#F8FAFC",
    marginBottom: 8,
    letterSpacing: -0.5,
    fontFamily: "Pharaoh",
  },
  subtitle: {
    fontSize: 16,
    color: "rgba(203, 213, 225, 0.8)",
    marginBottom: 40,
    textAlign: "center",
  },
  buttonContainer: {
    width: "100%",
    maxWidth: 350,
    gap: 16,
  },
  button: {
    height: 55,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
    borderWidth: 1,
  },
  primaryButton: {
    backgroundColor: "rgba(56, 189, 248, 0.15)",
    borderColor: "rgba(56, 189, 248, 0.6)",
  },
  secondaryButton: {
    backgroundColor: "rgba(71, 85, 105, 0.3)",
    borderColor: "rgba(71, 85, 105, 0.6)",
  },
  buttonText: {
    color: "#38BDF8",
    fontSize: 18,
    fontWeight: "600",
    letterSpacing: 0.5,
    textAlign: "center",
  },
  loadingText: {
    fontSize: 18,
    color: "#FFFFFF",
    textAlign: "center",
    marginTop: "50%",
  },
  footerContainer: {
    alignItems: "center",
    justifyContent: "center",
    marginTop: 20,
    marginBottom: 20,
  },
  footerText: {
    fontSize: 18,
    color: "#64748B",
    textAlign: "center",
    fontFamily: "Pharaoh",
  },
});
