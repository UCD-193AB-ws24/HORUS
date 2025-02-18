import * as Speech from 'expo-speech';
import { Button } from 'react-native';


export function StaticTextToSpeech({ input }: { input: string }) {
  return (
    <Button
        title="Press To Hear Speech"
        onPress={() => Speech.speak(input)}
    />  
  );
}