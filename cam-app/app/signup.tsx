import { useState } from "react";
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "@/lib/AuthContext";
import { Colors } from "@/constants/Colors";
import { useColorScheme } from "react-native";

export default function SignUp() {
  const { signUp } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? "light"];

  const handleSubmit = async () => {
    const { error } = await signUp(email, password);
    if (error) setErrorMsg(error.message);
    else router.replace("/signin");
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={[styles.container, { backgroundColor: colors.background }]}
    >
      <View style={styles.content}>
        <Text style={[styles.title, { color: colors.text }]}>
          Create Account
        </Text>
        <Text style={[styles.subtitle, { color: colors.icon }]}>
          Sign up to get started
        </Text>

        <View style={styles.inputContainer}>
          <TextInput
            placeholder="Email"
            placeholderTextColor={colors.icon}
            value={email}
            onChangeText={setEmail}
            style={[
              styles.input,
              {
                backgroundColor: colorScheme === "dark" ? "#2C2C2E" : "#F2F2F7",
                color: colors.text,
                borderColor: errorMsg ? "#FF3B30" : "transparent",
              },
            ]}
            autoCapitalize="none"
            keyboardType="email-address"
            autoComplete="email"
          />
          <TextInput
            placeholder="Password"
            placeholderTextColor={colors.icon}
            value={password}
            onChangeText={setPassword}
            style={[
              styles.input,
              {
                backgroundColor: colorScheme === "dark" ? "#2C2C2E" : "#F2F2F7",
                color: colors.text,
                borderColor: errorMsg ? "#FF3B30" : "transparent",
              },
            ]}
            secureTextEntry
            autoComplete="password-new"
          />
          {errorMsg && <Text style={styles.error}>{errorMsg}</Text>}
        </View>

        <TouchableOpacity
          style={[styles.button, { backgroundColor: colors.tint }]}
          onPress={handleSubmit}
        >
          <Text style={styles.buttonText}>Sign Up</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.signInButton}
          onPress={() => router.push("/signin")}
        >
          <Text style={[styles.signInText, { color: colors.tint }]}>
            Already have an account? Sign In
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 20,
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 32,
  },
  inputContainer: {
    marginBottom: 24,
    width: "80%",
  },
  input: {
    height: 50,
    borderRadius: 10,
    paddingHorizontal: 16,
    marginBottom: 16,
    fontSize: 16,
    borderWidth: 1,
  },
  error: {
    color: "#FF3B30",
    marginBottom: 16,
    fontSize: 14,
  },
  button: {
    height: 50,
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
    width: "80%",
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  signInButton: {
    alignItems: "center",
  },
  signInText: {
    fontSize: 16,
    fontWeight: "500",
  },
});
