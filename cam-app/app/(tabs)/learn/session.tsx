import { useCallback, useMemo, useRef, useState } from "react";
import { View, Text, Image, Switch } from "react-native";
import { useRouter } from "expo-router";
import tw from "twrnc";
import useAlphabetLesson from "@/hooks/useAlphabetLesson";
import LessonCamera from "@/components/LessonCamera";

const signImages: Record<string, number> = {
  a: require("@/assets/signs/a.png"),
  b: require("@/assets/signs/b.png"),
  c: require("@/assets/signs/c.png"),
  d: require("@/assets/signs/d.png"),
  e: require("@/assets/signs/e.png"),
  f: require("@/assets/signs/f.png"),
  g: require("@/assets/signs/g.png"),
  h: require("@/assets/signs/h.png"),
  i: require("@/assets/signs/i.png"),
  j: require("@/assets/signs/j.png"),
  k: require("@/assets/signs/k.png"),
  l: require("@/assets/signs/l.png"),
  m: require("@/assets/signs/m.png"),
  n: require("@/assets/signs/n.png"),
  o: require("@/assets/signs/o.png"),
  p: require("@/assets/signs/p.png"),
  q: require("@/assets/signs/q.png"),
  r: require("@/assets/signs/r.png"),
  s: require("@/assets/signs/s.png"),
  t: require("@/assets/signs/t.png"),
  u: require("@/assets/signs/u.png"),
  v: require("@/assets/signs/v.png"),
  w: require("@/assets/signs/w.png"),
  x: require("@/assets/signs/x.png"),
  y: require("@/assets/signs/y.png"),
  z: require("@/assets/signs/z.png"),
};

const CONGRATULATORY_MESSAGES = [
  "Good job!",
  "That was awesome!",
  "You're amazing!",
  "Perfect!",
  "Keep it up!",
  "You're a natural!",
  "Excellent work!",
];

export default function Session() {
  const router = useRouter();
  const { letters } = useAlphabetLesson();
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [congratulatoryMessage, setCongratulatoryMessage] = useState("");
  const [showImage, setShowImage] = useState(false);
  const lock = useRef(false);

  const currentLetter = useMemo(() => letters[index], [letters, index]);

  const handleDetect = useCallback(
    (letter: string) => {
      if (lock.current || letter !== currentLetter) return;
      lock.current = true;

      const nextScore = score + 50;
      const rand = Math.floor(Math.random() * CONGRATULATORY_MESSAGES.length);
      setCongratulatoryMessage(CONGRATULATORY_MESSAGES[rand]);

      if (index === letters.length - 1) {
        router.replace({
          pathname: "/(tabs)/learn/result",
          params: { score: String(nextScore) },
        });
        return;
      }

      setScore(nextScore);
      setIndex((i) => i + 1);
      setTimeout(() => {
        lock.current = false;
      }, 500);
    },
    [currentLetter, index, letters.length, router, score]
  );

  return (
    <View style={tw`flex-1 bg-white dark:bg-black p-4`}>
      <View style={tw`flex-row justify-end items-center mb-4`}>
        <Text style={tw`text-gray-800 dark:text-gray-100 mr-2`}>
          Show image
        </Text>
        <Switch
          value={showImage}
          onValueChange={setShowImage}
          trackColor={{ false: "#767577", true: "#81b0ff" }}
          thumbColor={showImage ? "#f5dd4b" : "#f4f3f4"}
        />
      </View>

      <View style={tw`self-center w-4/5 h-3/5 overflow-hidden rounded-xl`}>
        <LessonCamera onDetect={handleDetect} />
      </View>

      {showImage && (
        <View style={tw`mt-4 items-center`}>
          <Image
            source={signImages[currentLetter.toLowerCase()]}
            // source={require("@/assets/signs/a.png")}
            style={tw`w-24 h-24 rounded-lg`}
            resizeMode="contain"
          />
        </View>
      )}

      <View style={tw`mt-6 items-center`}>
        <Text
          style={tw`text-xl font-semibold text-gray-800 dark:text-gray-100`}
        >
          Sign the letter {currentLetter}
        </Text>
        {congratulatoryMessage !== "" && (
          <Text
            style={tw`text-lg font-bold text-green-600 dark:text-green-400 mt-4`}
          >
            {congratulatoryMessage}
          </Text>
        )}
        <Text
          style={tw`text-lg font-bold text-gray-800 dark:text-gray-100 mt-4`}
        >
          Score: {score}
        </Text>
      </View>
    </View>
  );
}
