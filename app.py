import streamlit as st
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from gtts import gTTS
import base64

# 🎯 Server URLs (Update if deployed)
EMOTION_SERVER_URL = "http://127.0.0.1:5001/predict"
COHERE_SERVER_URL = "http://127.0.0.1:5001/cohere_response"

# 🎙 Audio settings
FILENAME = "recorded_audio.wav"
DURATION = 7  # seconds
SAMPLE_RATE = 16000  # Hz

# 🎨 Page settings
st.set_page_config(page_title="Emotion-Based AI Chatbot", layout="centered")

# 🎨 Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .title {{
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-align: center;
            padding: 10px;
        }}
        .section {{
            text-align: center;
            font-weight: bold;
            color: white;
            padding: 10px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            margin: 10px 0;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 🌄 Set background image (Make sure image.png is in the same directory!)
set_background("image.png")

# 📌 Page header
st.markdown('<div class="title">🎙 Emotion-Based AI Chatbot</div>', unsafe_allow_html=True)
st.markdown("<div class='section'><h3>Speak into the microphone, and the AI will respond based on your emotion.</h3></div>", unsafe_allow_html=True)

# 🎵 Function to play AI-generated response
def speak_text(text):
    try:
        tts = gTTS(text)
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
    except Exception as e:
        st.markdown(f"<div class='section'>❌ TTS Error: {e}</div>", unsafe_allow_html=True)

# 🎙 Function to record audio
def record_audio():
    """Records audio and saves it as a WAV file."""
    st.markdown("<div class='section'>🎤 Recording...</div>", unsafe_allow_html=True)
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(FILENAME, SAMPLE_RATE, audio)
    st.markdown("<div class='section'>✅ Recording saved!</div>", unsafe_allow_html=True)

# 🔤 Function to convert speech to text
def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.AudioFile(FILENAME) as source:
        st.markdown("<div class='section'>📝 Converting speech to text...</div>", unsafe_allow_html=True)
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.markdown(f"<div class='section'>🗣 Recognized Text: {text}</div>", unsafe_allow_html=True)
            return text
        except sr.UnknownValueError:
            st.markdown("<div class='section'>⚠ Could not understand the audio</div>", unsafe_allow_html=True)
            return None
        except sr.RequestError:
            st.markdown("<div class='section'>❌ Speech-to-text service unavailable</div>", unsafe_allow_html=True)
            return None

# 😃 Function to detect emotion
def get_emotion():
    st.markdown("<div class='section'>📤 Detecting emotion...</div>", unsafe_allow_html=True)
    try:
        with open(FILENAME, "rb") as file:
            response = requests.post(EMOTION_SERVER_URL, files={"file": file})
        response_json = response.json()
        emotion = response_json.get("emotion", "unknown")
        st.markdown(f"<div class='section'>🔊 Predicted Emotion: {emotion}</div>", unsafe_allow_html=True)
        return emotion
    except Exception as e:
        st.markdown(f"<div class='section'>❌ Error: {e}</div>", unsafe_allow_html=True)
        return None

# 🤖 Function to get AI chatbot response
def get_cohere_response(text, emotion):
    st.markdown("<div class='section'>📤 Sending to AI...</div>", unsafe_allow_html=True)
    data = {"text": text, "emotion": emotion}
    try:
        response = requests.post(COHERE_SERVER_URL, json=data)
        if response.status_code == 200:
            ai_response = response.json().get("response", "No response received.")
            st.markdown(f"<div class='section'>🤖 AI: {ai_response}</div>", unsafe_allow_html=True)
            return ai_response
        else:
            st.markdown(f"<div class='section'>❌ Cohere Error: {response.json()}</div>", unsafe_allow_html=True)
            return None
    except Exception as e:
        st.markdown(f"<div class='section'>❌ Error: {e}</div>", unsafe_allow_html=True)
        return None

# 🎤 Main button to trigger recording and chatbot interaction
st.markdown(
    "<div style='text-align: center;'>",
    unsafe_allow_html=True,
)
if st.button("🎤 Start Recording"):
    record_audio()
    text = convert_speech_to_text()
    if text:
        emotion = get_emotion()
        if emotion:
            response_text = get_cohere_response(text, emotion)
            if response_text:
                speak_text(response_text)
st.markdown("</div>", unsafe_allow_html=True)
