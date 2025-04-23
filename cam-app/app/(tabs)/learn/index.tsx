import { View, Pressable, Text } from "react-native";
import { useRouter } from "expo-router";
import tw from "twrnc";

export default function AlphabetLessonStart() {
  const router = useRouter();

  return (
    <View style={tw`flex-1 justify-center items-center bg-white dark:bg-black`}>
      <Pressable
        accessibilityRole="button"
        onPress={() => router.push("/(tabs)/learn/session")}
        style={tw`bg-[#0a7ea4] px-6 py-4 rounded-lg`}
      >
        <Text style={tw`text-white text-lg font-semibold`}>
          Click here to start learning ASL
        </Text>
      </Pressable>
    </View>
  );
}
