from pyht import Client 
from pyht.client import TTSOptions
from playsound import playsound
import os
import whisper

client = Client(
    user_id = "satC3ZeLC7Vkcdc155yjcl7laFu2",
    api_key = "345a693e11304d748f702f67de03bfec",
)

options = TTSOptions(voice="s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json")
text_to_speak = input("Enter what you would like to be said: ")
with open("output.wav", 'wb') as audio_file:
    for chunk in client.tts(text_to_speak, options, voice_engine = 'PlayDialog', protocol='http'):
        audio_file.write(chunk)

model = whisper.load_model("tiny.en")
result = model.transcribe("output.wav")
print(result["text"])
os.remove("output.wav")

