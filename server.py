import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import io
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import cohere
import requests
from gtts import gTTS
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load API keys from Hugging Face Secrets
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# Load Emotion Detection Model
MODEL_PATH = "my_trained_model.pth"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

LABEL_MAP = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "disgust", 6: "surprised"}

@app.route("/predict", methods=["POST"])
def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_data, samplerate = sf.read(io.BytesIO(file.read()), dtype="float32")
    
    inputs = processor(audio_data, sampling_rate=samplerate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = torch.argmax(logits, dim=-1).item()
    emotion = LABEL_MAP.get(predicted_label, "unknown")

    return jsonify({"emotion": emotion})

# Initialize Cohere API
co = cohere.Client(COHERE_API_KEY)

@app.route("/cohere_response", methods=["POST"])
def get_cohere_response():
    data = request.json
    user_text = data.get("text", "")
    user_emotion = data.get("emotion", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = co.chat(
            model="command-r-plus",
            message=f"User is feeling {user_emotion}. They said: {user_text}"
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")
    
    audio = AudioSegment.from_file("output.mp3")
    fast_audio = audio.speedup(playback_speed=1.25)
    fast_audio.export("output_fast.mp3", format="mp3")
    
    return send_file("output_fast.mp3", mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
