// app/_layout.tsx

"use client";

import React, { useEffect, useState, createContext } from "react";
import {
  DarkTheme,
  DefaultTheme,
  ThemeProvider,
} from "@react-navigation/native";
import { useFonts } from "expo-font";
import { Stack } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import "react-native-reanimated";
import { TouchableOpacity, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";

import { Colors } from "@/constants/Colors";
import { AuthProvider, useAuth } from "@/lib/AuthContext";

export const ThemeContext = createContext<{
  theme: "light" | "dark";
  toggleTheme: () => void;
}>({
  theme: "light",
  toggleTheme: () => {},
});

// Prevent the splash screen from auto-hiding until fonts load
SplashScreen.preventAutoHideAsync();

function RootStack() {
  const { session } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!session) {
      router.replace("/signin");
    }
  }, [session]);

  return (
    <Stack screenOptions={{ headerShown: false }}>
      {!session ? (
        <>
          <Stack.Screen name="signin" />
          <Stack.Screen name="signup" />
        </>
      ) : (
        <>
          <Stack.Screen name="(tabs)" />
          <Stack.Screen name="+not-found" />
        </>
      )}
    </Stack>
  );
}

export default function RootLayout() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [fontsLoaded] = useFonts({
    SpaceMono: require("../assets/fonts/SpaceMono-Regular.ttf"),
  });

  useEffect(() => {
    if (fontsLoaded) {
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded]);

  if (!fontsLoaded) {
    return null;
  }

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <ThemeProvider value={theme === "dark" ? DarkTheme : DefaultTheme}>
        <AuthProvider>
          <RootStack />
        </AuthProvider>

        {/* <TouchableOpacity
          style={[
            styles.themeToggle,
            {
              backgroundColor:
                theme === "dark"
                  ? Colors.dark.background
                  : Colors.light.background,
            },
          ]}
          onPress={toggleTheme}
        >
          <Ionicons
            name={theme === "dark" ? "moon" : "sunny"}
            size={24}
            color={theme === "dark" ? Colors.dark.text : Colors.light.text}
          />
        </TouchableOpacity> */}
      </ThemeProvider>
    </ThemeContext.Provider>
  );
}

const styles = StyleSheet.create({
  themeToggle: {
    position: "absolute",
    top: 50,
    right: 20,
    zIndex: 1000,
    padding: 8,
    borderRadius: 20,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
});
