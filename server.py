import os
import gdown  # Install with `pip install gdown`
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Google Drive file ID for the model
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/1jHqUsguayTcoyxW1Ckqu8k4uLIlEXzai/view?usp=sharing"  # Replace this with your actual file ID
MODEL_PATH = "my_trained_model.pth"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):  # Only download if not already present
        print("🔽 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")

# Ensure model is available
download_model()

# Load the trained emotion detection model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=7, state_dict=torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()  # Set model to evaluation mode

# Emotion label mapping
LABEL_MAP = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "disgust", 6: "surprised"}

@app.route("/predict", methods=["POST"])
def predict_emotion():
    """Handles audio file input and returns predicted emotion."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_data, samplerate = sf.read(io.BytesIO(file.read()), dtype="float32")

    # Preprocess audio for model
    inputs = processor(audio_data, sampling_rate=samplerate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = torch.argmax(logits, dim=-1).item()
    emotion = LABEL_MAP.get(predicted_label, "unknown")

    return jsonify({"emotion": emotion})

# Initialize Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Use environment variable
if not COHERE_API_KEY:
    raise ValueError("❌ Cohere API key missing. Check .env file.")

co = cohere.Client(COHERE_API_KEY)

@app.route("/cohere_response", methods=["POST"])
def get_cohere_response():
    """Receives text and generates a response using Cohere."""
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)  # Allow external access
