import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import io
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Model Download
MODEL_PATH = "my_trained_model.pth"
GOOGLE_DRIVE_FILE_ID = "1jHqUsguayTcoyxW1Ckqu8k4uLIlEXzai"  # Replace with actual ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

download_model()

# Load Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=7, state_dict=torch.load(MODEL_PATH, map_location="cpu")
)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
