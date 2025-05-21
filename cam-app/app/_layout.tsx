"use client";

import React, { useEffect, useState, createContext, useRef } from "react";
import {
  DarkTheme,
  DefaultTheme,
  ThemeProvider,
} from "@react-navigation/native";
import { useFonts } from "expo-font";
import { Stack, useRouter } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import "react-native-reanimated";
import { StyleSheet } from "react-native";

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
  const hasRedirected = useRef(false);

  useEffect(() => {
    if (!session && !hasRedirected.current) {
      hasRedirected.current = true;
      setTimeout(() => {
        router.replace("/signin");
      }, 0);
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
