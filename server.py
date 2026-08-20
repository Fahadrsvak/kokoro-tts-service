import asyncio
import json
import os
import logging
import websockets
import numpy as np
from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

# Initialize Kokoro ONNX Model
logging.info("Initializing Kokoro ONNX model into memory...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
logging.info("Kokoro ONNX engine ready.")

async def tts_handler(websocket):
    client_ip = websocket.remote_address[0]
    logging.info(f"Client connected: {client_ip}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logging.warning("Received invalid JSON payload.")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))
                continue

            text = data.get("text", "").strip()
            voice = data.get("voice", "af_sky")
            speed = float(data.get("speed", 1.0))
            lang = data.get("lang", "en-us")

            if not text:
                continue

            logging.info(f"Synthesizing for {client_ip} | Voice: {voice} | Length: {len(text)} chars")

            try:
                # Stream audio chunks as sentences finish generating
                stream = kokoro.create_stream(
                    text=text,
                    voice=voice,
                    speed=speed,
                    lang=lang
                )

                async for samples, sample_rate in stream:
                    # Convert float32 array [-1.0, 1.0] to 16-bit PCM binary
                    pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                    pcm_bytes = pcm_samples.tobytes()
                    
                    # Stream raw binary audio frame directly to client
                    await websocket.send(pcm_bytes)

                # Send lightweight EOF signal to mark completion
                await websocket.send(json.dumps({"status": "EOF", "sample_rate": 24000}))

            except Exception as e:
                logging.error(f"Error during synthesis stream: {e}")
                await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Client disconnected: {client_ip}")

async def main():
    port = int(os.environ.get("PORT", 8880))
    async with websockets.serve(tts_handler, "0.0.0.0", port):
        logging.info(f"🚀 Kokoro WebSocket TTS running on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())