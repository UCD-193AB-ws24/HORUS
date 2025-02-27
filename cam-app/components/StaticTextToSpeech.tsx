import * as Speech from 'expo-speech';
import { useState } from 'react';
import { Button, TextInput } from 'react-native';

export function StaticTextToSpeech() {
  const [input, setInput] = useState<string>("");
  
  return (
    <>
      <TextInput
        style={{ height: 40, borderColor: 'gray', borderWidth: 1 }}
        onChangeText={text => setInput(text)}
        value={input}
        />
      <Button
        title="Press To Hear Speech"
        onPress={() => {
          Speech.speak(input)
          console.log("Speech Spoken: ", input)
        }}
      /> 
    </> 
  );
}