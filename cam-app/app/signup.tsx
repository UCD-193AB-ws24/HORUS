import { useState, useEffect } from "react";
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  StatusBar,
  Image,
  ScrollView,
  Keyboard,
} from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "@/lib/AuthContext";
import { Ionicons } from "@expo/vector-icons";

export default function SignUp() {
  const { signUp } = useAuth();
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (Platform.OS !== "web") return;

    const style = document.createElement("style");
    style.textContent = `
      input:-webkit-autofill,
      input:-webkit-autofill:hover,
      input:-webkit-autofill:focus,
      input:-internal-autofill-selected {
        -webkit-box-shadow: 0 0 0px 1000px rgba(15,23,42,0.3) inset !important;
        -webkit-text-fill-color: #FFFFFF !important;
        transition: background-color 9999s ease-out, color 9999s ease-out;
      }
    `;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const handleSubmit = async () => {
    const { error } = await signUp(email, password);
    if (error) setErrorMsg(error.message);
    else router.replace("/signin");
  };

  const content = (
    <>
      <StatusBar barStyle="light-content" />
      <View style={styles.backgroundGradient} />

      <View style={styles.splitContainer}>
        <View style={styles.formSide}>
          <View style={styles.formContainer}>
            <Text style={styles.title}>Join Us</Text>
            <Text style={styles.subtitle}>
              Register for a free account today
            </Text>

            <View style={styles.inputContainer}>
              <Field
                icon="mail-outline"
                placeholder="Email"
                value={email}
                onChangeText={setEmail}
                textContentType="emailAddress"
                keyboardType="email-address"
                secure={false}
              />

              <Field
                icon="lock-closed-outline"
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                textContentType="password"
                secure={true}
              />

              {errorMsg && <Text style={styles.error}>{errorMsg}</Text>}
            </View>

            <TouchableOpacity style={styles.button} onPress={handleSubmit}>
              <Text style={styles.buttonText}>Create Account</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.signInButton}
              onPress={() => router.push("/signin")}
            >
              <Text style={styles.signInText}>
                Already have an account?{" "}
                <Text style={styles.signInTextHighlight}>Sign In</Text>
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.brandingSide}>
          <View style={styles.brandingContent}>
            <Image
              source={require("@/assets/images/horus.png")}
              style={styles.logo}
              resizeMode="contain"
            />
            <Text style={styles.appTitle}>HORUS</Text>
          </View>

          <View style={styles.footerContainer}>
            <Text style={styles.tagline}>
              American Sign Language Translation App
            </Text>
            <Text style={styles.footerText}>Made by group6 LLC</Text>
          </View>
        </View>
      </View>
    </>
  );

  if (Platform.OS === "web") {
    return <View style={styles.container}>{content}</View>;
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      style={styles.container}
      keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 0}
    >
      <ScrollView
        contentContainerStyle={{ flexGrow: 1 }}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {content}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function Field({
  icon,
  placeholder,
  value,
  onChangeText,
  textContentType,
  keyboardType = "default",
  secure,
}: {
  icon: any;
  placeholder: string;
  value: string;
  onChangeText: (t: string) => void;
  textContentType: any;
  keyboardType?: "default" | "email-address";
  secure: boolean;
}) {
  return (
    <View style={styles.inputWrapper}>
      <Ionicons
        name={icon}
        size={20}
        color="#94A3B8"
        style={styles.inputIcon}
      />
      <TextInput
        placeholder={placeholder}
        placeholderTextColor="#94A3B8"
        value={value}
        onChangeText={onChangeText}
        style={styles.input}
        textContentType={textContentType}
        keyboardType={keyboardType}
        secureTextEntry={secure}
        autoCapitalize="none"
        autoComplete="off"
        autoCorrect={false}
        spellCheck={false}
        caretHidden={false}
        selectionColor="#10B981"
        underlineColorAndroid="transparent"
      />
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
  splitContainer: {
    flex: 1,
    flexDirection: "row",
  },
  formSide: { flex: 1.2, justifyContent: "center", alignItems: "center" },
  formContainer: { width: "75%", padding: 40 },
  title: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#F8FAFC",
    marginBottom: 8,
    fontFamily: "Pharaoh",
  },
  subtitle: { fontSize: 16, color: "rgba(203,213,225,0.8)", marginBottom: 20 },

  inputContainer: { gap: 16, width: "100%", marginBottom: 20 },

  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 16,
    backgroundColor: "rgba(15,23,42,0.3)",
    borderWidth: 1,
    borderColor: "rgba(16,185,129,0.5)",
    overflow: "hidden",
    paddingHorizontal: 4,
  },
  inputIcon: { marginLeft: 16, opacity: 0.8 },
  input: {
    flex: 1,
    height: 55,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#FFFFFF",
    backgroundColor: "transparent",
  },

  error: { color: "#F87171", fontSize: 14 },

  button: {
    height: 55,
    borderRadius: 16,
    backgroundColor: "rgba(16,185,129,0.15)",
    borderWidth: 1,
    borderColor: "rgba(16,185,129,0.6)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  buttonText: { color: "#10B981", fontSize: 18, fontWeight: "600" },

  signInButton: { alignSelf: "center" },
  signInText: { fontSize: 15, color: "rgba(148,163,184,0.9)" },
  signInTextHighlight: { color: "#10B981", fontWeight: "600" },

  brandingSide: {
    flex: 1,
    justifyContent: "space-between",
    alignItems: "center",
    paddingBottom: 30,
  },
  brandingContent: { flex: 1, justifyContent: "center", alignItems: "center" },
  footerContainer: { alignItems: "center", paddingBottom: 20 },
  tagline: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
    textAlign: "center",
    maxWidth: "80%",
    marginBottom: 10,
    fontFamily: "Pharaoh",
  },
  footerText: { fontSize: 12, color: "#64748B", fontFamily: "Pharaoh" },

  logo: { width: 200, height: 200, marginBottom: 10 },
  appTitle: {
    color: "#FFFFFF",
    fontSize: 36,
    fontWeight: "bold",
    letterSpacing: 4,
    textAlign: "center",
    textTransform: "uppercase",
  },
});
