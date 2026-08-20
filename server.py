import asyncio
import json
import os
import re
import logging
import websockets
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

# 1. Configure ONNX Multi-Threading Options for ARM CPUs
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2  # Match available CPU cores
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Initialize Kokoro ONNX Model
logging.info("Initializing Kokoro ONNX model into memory...")
kokoro = Kokoro(
    MODEL_PATH, 
    VOICES_PATH,
    session_options=session_options
)

# 2. Warmup Model to pre-allocate memory buffers before first user request
try:
    logging.info("Warming up Kokoro execution engine...")
    _ = list(kokoro.create_stream("Warmup text.", voice="af_heart", speed=1.0, lang="en-us"))
    logging.info("Kokoro ONNX engine ready and warmed up.")
except Exception as err:
    logging.warning(f"Warmup warning: {err}")

def split_into_sentences(text: str):
    """Splits text on punctuation boundaries to synthesize sentence-by-sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

async def tts_handler(websocket):
    client_ip = websocket.remote_address[0]
    logging.info(f"Client connected: {client_ip}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logging.warning("Received invalid JSON payload.")
                await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON format"}))
                continue

            text = data.get("text", "").strip()
            voice = data.get("voice", "af_heart")
            speed = float(data.get("speed", 1.0))
            lang = data.get("lang", "en-us")

            if not text:
                continue

            logging.info(f"Synthesizing for {client_ip} | Voice: {voice} | Length: {len(text)} chars")

            # Send start signal matching test client
            await websocket.send(json.dumps({"type": "start", "sample_rate": 24000}))

            try:
                # Break down input into sentence chunks to minimize initial TTFA
                sentences = split_into_sentences(text)

                for sentence in sentences:
                    stream = kokoro.create_stream(
                        text=sentence,
                        voice=voice,
                        speed=speed,
                        lang=lang
                    )

                    async for samples, sample_rate in stream:
                        pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                        pcm_bytes = pcm_samples.tobytes()
                        await websocket.send(pcm_bytes)

                # Send completion signal matching test client
                await websocket.send(json.dumps({"type": "end", "status": "EOF", "sample_rate": 24000}))

            except Exception as e:
                logging.error(f"Error during synthesis stream: {e}")
                await websocket.send(json.dumps({"type": "error", "error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Client disconnected: {client_ip}")

async def main():
    port = 6007
    async with websockets.serve(tts_handler, "0.0.0.0", port):
        logging.info(f"🚀 Kokoro WebSocket TTS running directly on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())