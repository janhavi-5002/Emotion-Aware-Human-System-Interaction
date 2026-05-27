from voice_emotion import start_voice_detection, get_voice_emotion
import time

print("Voice Emotion Detection Start ho rahi hai...")
print("Microphone mein bolo - Ctrl+C dabao band karne ke liye\n")

start_voice_detection()

try:
    while True:
        emotion = get_voice_emotion()
        print(f"Current Voice Emotion: {emotion}")
        time.sleep(3)
except KeyboardInterrupt:
    print("\nVoice detection band ho gayi")