import { StyleSheet, Image, TextInput, Text, Button} from "react-native";
import React from 'react';
import ParallaxScrollView from "@/components/ParallaxScrollView";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';

export default function HelpScreen() {
    const [signed, onChangeSigned] = React.useState('');
    const [translated, onChangeSignTranslated] = React.useState('');

    const sendFormToServer = async () => {
        try {
            let responseFetch = await fetch('http://127.0.0.1:8000/send_help_form/', {
                method: 'POST',
                body: JSON.stringify({"signed": signed, "translated": translated}),
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
            });
            if (!responseFetch.ok) {
                const errorText = await responseFetch.text();
                console.error('[ERROR] Server response:', errorText);
                throw new Error(errorText);
            }
        } catch (error) {
            console.error('[ERROR] Error sending form:', error);
        }
        onChangeSigned('');
        onChangeSignTranslated('');
    };


  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: "#F5F5F5", dark: "#1D3D47" }}
      headerImage={
        <Image
          source={require("@/assets/images/tech-support-icon.jpg")}
          style={styles.landingLogo}
          resizeMode="contain"
        />
      }
    >
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Report a Problem</ThemedText>
      </ThemedView>
      <Text style={styles.helpText}>Type in the box what sign you tried to sign:</Text>
      <SafeAreaProvider>
        <SafeAreaView>
            <TextInput
            style={styles.input}
            onChangeText={onChangeSigned}
            value={signed}
            />
        </SafeAreaView>
      </SafeAreaProvider>
      <Text style={styles.helpText}>Type in the box what it said you signed:</Text>
      <SafeAreaProvider>
        <SafeAreaView>
            <TextInput
            style={styles.input}
            onChangeText={onChangeSignTranslated}
            value={translated}
            />
        </SafeAreaView>
      </SafeAreaProvider>
      <Button title="Submit" onPress={sendFormToServer} color="black"/>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: "#808080",
    bottom: -90,
    left: -35,
    position: "absolute",
  },
  titleContainer: {
    flexDirection: "row",
    gap: 8,
  },
  landingLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: "static",
  },
  input: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
  },
  helpText: {
    fontSize: 24,
    height: 60,
    padding: 1,
  },
});
