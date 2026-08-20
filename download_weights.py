import os
import urllib.request

WEIGHTS_DIR = "weights"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

model_path = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
voices_path = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

if not os.path.exists(model_path):
    print("Downloading Kokoro-82M ONNX model...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Downloaded model successfully.")

if not os.path.exists(voices_path):
    print("Downloading Kokoro voices binary...")
    urllib.request.urlretrieve(VOICES_URL, voices_path)
    print("Downloaded voices successfully.")