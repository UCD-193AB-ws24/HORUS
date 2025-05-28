import React, { useState } from "react";
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
} from "react-native";
import { TabBarIcon } from "./TabBarIcon";
import { ThemedText } from "../ThemedText";
import { useRouter } from "expo-router";
import { Colors } from "@/constants/Colors";
import { useColorScheme } from "@/hooks/useColorScheme";

const { width: screenWidth } = Dimensions.get("window");

interface MenuItem {
  label: string;
  route: string;
  icon: string;
  iconFocused: string;
}

const menuItems: MenuItem[] = [
  {
    label: "Home",
    route: "/(tabs)/",
    icon: "home-outline",
    iconFocused: "home",
  },
  {
    label: "Camera",
    route: "/(tabs)/camera",
    icon: "camera-outline",
    iconFocused: "camera",
  },
  {
    label: "Learn",
    route: "/(tabs)/learn",
    icon: "school-outline",
    iconFocused: "school",
  },
];

export function HamburgerMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const [animation] = useState(new Animated.Value(0));
  const router = useRouter();
  const colorScheme = useColorScheme();

  const toggleMenu = () => {
    const toValue = isOpen ? 0 : 1;
    setIsOpen(!isOpen);

    Animated.timing(animation, {
      toValue,
      duration: 200,
      useNativeDriver: true,
    }).start();
  };

  const navigateToRoute = (route: string) => {
    router.replace(route as any);
    toggleMenu();
  };

  const dropdownTranslateY = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [-10, 0],
  });

  const dropdownOpacity = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.hamburgerButton}
        onPress={toggleMenu}
        activeOpacity={0.7}
      >
        <TabBarIcon
          name={isOpen ? "close" : "menu"}
          color="#FFFFFF"
          size={24}
        />
      </TouchableOpacity>

      {isOpen && (
        <Animated.View
          style={[
            styles.dropdown,
            {
              opacity: dropdownOpacity,
              transform: [{ translateY: dropdownTranslateY }],
            },
          ]}
        >
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={item.route}
              style={[
                styles.menuItem,
                index === menuItems.length - 1 && styles.lastMenuItem,
              ]}
              onPress={() => navigateToRoute(item.route)}
              activeOpacity={0.7}
            >
              <TabBarIcon name={item.icon as any} color="#FFFFFF" size={20} />
              <ThemedText style={styles.menuText}>{item.label}</ThemedText>
            </TouchableOpacity>
          ))}
        </Animated.View>
      )}

      {isOpen && (
        <TouchableOpacity
          style={styles.overlay}
          onPress={toggleMenu}
          activeOpacity={1}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: "relative",
    zIndex: 1000,
  },
  hamburgerButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  dropdown: {
    position: "absolute",
    top: 50,
    right: 0,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    borderRadius: 12,
    paddingVertical: 8,
    minWidth: 140,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 8,
    backdropFilter: "blur(10px)",
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255, 255, 255, 0.1)",
  },
  lastMenuItem: {
    borderBottomWidth: 0,
  },
  menuText: {
    marginLeft: 12,
    fontSize: 16,
    fontWeight: "500",
    color: "#FFFFFF",
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: -screenWidth,
    width: screenWidth * 2,
    height: 1000,
    zIndex: -1,
  },
});
