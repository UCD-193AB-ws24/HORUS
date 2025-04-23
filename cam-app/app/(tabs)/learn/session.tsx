import { useCallback, useMemo, useRef, useState } from "react";
import { View, Text } from "react-native";
import { useRouter } from "expo-router";
import tw from "twrnc";
import useAlphabetLesson from "@/hooks/useAlphabetLesson";
import LessonCamera from "@/components/LessonCamera";

export default function AlphabetLessonSession() {
  const router = useRouter();
  const { letters } = useAlphabetLesson();
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const lock = useRef(false);

  const currentLetter = useMemo(() => letters[index], [letters, index]);

  const handleDetect = useCallback(
    (letter: string) => {
      if (lock.current || letter !== currentLetter) return;

      lock.current = true;
      const nextScore = score + 50;

      if (index === letters.length - 1) {
        router.replace({
          pathname: "/(tabs)/learn/result",
          params: { score: String(nextScore) },
        });
        return;
      }

      setScore(nextScore);
      setIndex((prev) => prev + 1);
      setTimeout(() => (lock.current = false), 500);
    },
    [currentLetter, index, letters.length, router, score]
  );

  return (
    <View style={tw`flex-1 bg-white dark:bg-black p-4`}>
      <View style={tw`flex-row justify-end`}>
        <Text style={tw`text-lg font-bold text-gray-800 dark:text-gray-100`}>
          {score}
        </Text>
      </View>

      <View style={tw`self-center w-4/5 h-3/5 overflow-hidden rounded-xl`}>
        <LessonCamera onDetect={handleDetect} />
      </View>

      <View style={tw`mt-6 items-center`}>
        <Text
          style={tw`text-xl font-semibold text-gray-800 dark:text-gray-100`}
        >
          Sign the letter {currentLetter}
        </Text>
      </View>
    </View>
  );
}
