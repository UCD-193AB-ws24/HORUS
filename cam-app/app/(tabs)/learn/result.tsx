import { View, Text, Pressable } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import tw from "twrnc";

export default function AlphabetLessonResult() {
  const { score } = useLocalSearchParams<{ score?: string }>();
  const router = useRouter();
  const points = Number(score ?? 0);

  return (
    <View style={tw`flex-1 justify-center items-center bg-white dark:bg-black`}>
      <Text
        style={tw`text-2xl font-bold mb-8 text-gray-800 dark:text-gray-100`}
      >
        Great job – you scored {points} points!
      </Text>
      <Pressable
        accessibilityRole="button"
        onPress={() => router.replace("/(tabs)/learn")}
        style={tw`bg-blue-600 px-6 py-4 rounded-lg`}
      >
        <Text style={tw`text-white text-lg font-semibold`}>Try again</Text>
      </Pressable>
    </View>
  );
}
