import {
  View,
  Text,
  Pressable,
  StyleSheet,
  StatusBar,
  Image,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

export default function AlphabetLessonResult() {
  const { score } = useLocalSearchParams<{ score?: string }>();
  const router = useRouter();
  const points = Number(score ?? 0);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.backgroundGradient} />

      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
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

            <View style={styles.mainContent}>
              <Text style={styles.title}>Lesson Complete</Text>
              <Text style={styles.scoreText}>
                Great job – you scored {points} points!
              </Text>

              <Pressable
                accessibilityRole="button"
                onPress={() => router.replace("/(tabs)/learn")}
                style={styles.button}
              >
                <Text style={styles.buttonText}>Try again</Text>
              </Pressable>
            </View>

            <View style={styles.footerContainer}>
              <Text style={styles.footerText}>
                Keep practicing to improve your skills
              </Text>
            </View>
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0A1128",
  },
  scrollContent: {
    flexGrow: 1,
  },
  backgroundGradient: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "#0A1128",
  },
  contentContainer: {
    flex: 1,
    justifyContent: "space-between",
    padding: 20,
  },
  headerSection: {
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 40,
  },
  logo: {
    width: 150,
    height: 150,
    marginBottom: 10,
  },
  appTitle: {
    color: "#FFFFFF",
    fontSize: 36,
    fontWeight: "bold",
    letterSpacing: 4,
    textAlign: "center",
    textTransform: "uppercase",
    fontFamily: "Pharaoh",
  },
  mainContent: {
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#F8FAFC",
    marginBottom: 16,
    letterSpacing: -0.5,
    fontFamily: "Pharaoh",
    textAlign: "center",
  },
  scoreText: {
    fontSize: 22,
    color: "#38BDF8",
    marginBottom: 40,
    textAlign: "center",
    fontWeight: "600",
  },
  button: {
    height: 55,
    borderRadius: 16,
    backgroundColor: "rgba(56, 189, 248, 0.15)",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 24,
    borderWidth: 1,
    borderColor: "rgba(56, 189, 248, 0.6)",
    marginTop: 16,
    minWidth: 180,
  },
  buttonText: {
    color: "#38BDF8",
    fontSize: 18,
    fontWeight: "600",
    letterSpacing: 0.5,
  },
  footerContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingBottom: 20,
  },
  footerText: {
    fontSize: 14,
    color: "#64748B",
    textAlign: "center",
    fontFamily: "Pharaoh",
  },
});
