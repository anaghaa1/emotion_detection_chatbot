import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import speech_recognition as sr  # Speech-to-text

# Server URLs (Ensure Flask is running)
EMOTION_SERVER_URL = "http://127.0.0.1:5001/predict"
COHERE_SERVER_URL = "http://127.0.0.1:5001/cohere_response"

# Audio File Name
FILENAME = "recorded_audio.wav"

# Recording settings
DURATION = 7  # seconds
SAMPLE_RATE = 16000  # Hz

def record_audio():
    """Records audio and saves it as a WAV file."""
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  
    wav.write(FILENAME, SAMPLE_RATE, audio)
    print("✅ Recording saved!")

def convert_speech_to_text():
    """Converts recorded audio to text using Google Speech-to-Text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(FILENAME) as source:
        print("📝 Converting speech to text...")
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("🗣 Recognized Text:", text)
            return text
        except sr.UnknownValueError:
            print("⚠ Could not understand the audio")
            return None
        except sr.RequestError:
            print("❌ Speech-to-text service unavailable")
            return None

def get_emotion():
    """Sends audio to the server for emotion detection."""
    print("📤 Sending audio to server...")
    try:
        with open(FILENAME, "rb") as file:
            response = requests.post(EMOTION_SERVER_URL, files={"file": file})
        
        if response.status_code == 200:
            response_json = response.json()
            emotion = response_json.get("emotion", "unknown")
            print("🔊 Predicted Emotion:", emotion)
            return emotion
        else:
            print("❌ Error from emotion detection server:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("❌ Failed to reach the emotion detection server:", e)
        return None
    except requests.exceptions.JSONDecodeError:
        print("❌ Server returned an invalid response (not JSON).")
        return None

def get_cohere_response(text, emotion):
    """Sends the text and emotion to Cohere for a response."""
    if not text or not emotion:
        print("⚠ Missing text or emotion, skipping Cohere request.")
        return

    print("📤 Sending text and emotion to Cohere...")
    data = {"text": text, "emotion": emotion}

    try:
        response = requests.post(COHERE_SERVER_URL, json=data)

        if response.status_code == 200:
            response_json = response.json()
            print("🤖 Cohere Response:", response_json.get("response", "No response received."))
        else:
            print("❌ Error from Cohere server:", response.text)
    except requests.exceptions.RequestException as e:
        print("❌ Failed to reach Cohere server:", e)

if __name__ == "__main__":
    record_audio()
    text_output = convert_speech_to_text()
    
    if text_output:
        emotion = get_emotion()
        if emotion:
            get_cohere_response(text_output, emotion)
