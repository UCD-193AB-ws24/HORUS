/**
 * Learn more about light and dark modes:
 * https://docs.expo.dev/guides/color-schemes/
 */

import { useContext } from "react";
import { Colors } from "@/constants/Colors";
import { ThemeContext } from "@/app/_layout";

type Theme = "light" | "dark";

export function useThemeColor(
  props: { light?: string; dark?: string },
  colorName: keyof typeof Colors.light & keyof typeof Colors.dark
) {
  const { theme } = useContext(ThemeContext);
  const colorFromProps = props[theme as Theme];

  if (colorFromProps) {
    return colorFromProps;
  }
  return Colors[theme as Theme][colorName];
}
