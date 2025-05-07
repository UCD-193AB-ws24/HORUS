import { View, Pressable, Text } from "react-native";
import { useRouter } from "expo-router";
import tw from "twrnc";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/AuthContext";
import useAlphabetLesson from "@/hooks/useAlphabetLesson";

export default function AlphabetLessonStart() {
  const router = useRouter();
  const { getProgress, user } = useAuth();
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
        // Only show continue option if there's actual progress (not at the beginning)
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

  const startNewSession = () => {
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
      <View
        style={tw`flex-1 justify-center items-center bg-white dark:bg-black`}
      >
        <Text style={tw`text-gray-800 dark:text-gray-100 text-lg`}>
          Loading...
        </Text>
      </View>
    );
  }

  return (
    <View style={tw`flex-1 justify-center items-center bg-white dark:bg-black`}>
      <View style={tw`w-4/5 max-w-sm`}>
        {savedIndex !== null && (
          <Pressable
            accessibilityRole="button"
            onPress={continueSession}
            style={tw`bg-[#0a7ea4] px-6 py-4 rounded-lg mb-4 w-full`}
          >
            <Text style={tw`text-white text-lg font-semibold text-center`}>
              Continue from letter {letters[savedIndex]}
            </Text>
          </Pressable>
        )}

        <Pressable
          accessibilityRole="button"
          onPress={startNewSession}
          style={tw`${
            savedIndex !== null ? "bg-gray-600" : "bg-[#0a7ea4]"
          } px-6 py-4 rounded-lg w-full`}
        >
          <Text style={tw`text-white text-lg font-semibold text-center`}>
            {savedIndex !== null
              ? "Start from beginning"
              : "Click here to start learning ASL"}
          </Text>
        </Pressable>
      </View>
    </View>
  );
}
