import asyncio
import json
import os
import re
import logging
import concurrent.futures
import websockets
import numpy as np
from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

logging.info("Initializing Kokoro ONNX model into memory...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

# Thread pool dedicated to CPU-heavy synthesis without freezing asyncio
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Warmup both model and phonemizer
try:
    logging.info("Warming up execution pipeline...")
    _ = list(kokoro.create_stream("Hello", voice="af_heart", speed=1.0, lang="en-us"))
    logging.info("Kokoro ONNX engine ready and pre-warmed.")
except Exception as err:
    logging.warning(f"Warmup warning: {err}")

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_chunks_blocking(sentence, voice, speed, lang):
    """Runs inside worker thread to avoid blocking asyncio event loop."""
    chunks = []
    stream = kokoro.create_stream(text=sentence, voice=voice, speed=speed, lang=lang)
    for samples, sample_rate in stream:
        pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        chunks.append(pcm_samples.tobytes())
    return chunks

async def tts_handler(websocket):
    client_ip = websocket.remote_address[0]
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON format"}))
                continue

            text = data.get("text", "").strip()
            voice = data.get("voice", "af_heart")
            speed = float(data.get("speed", 1.0))
            lang = data.get("lang", "en-us")

            if not text:
                continue

            # 1. Immediately acknowledge request
            await websocket.send(json.dumps({"type": "start", "sample_rate": 24000}))

            loop = asyncio.get_running_loop()
            sentences = split_into_sentences(text)

            try:
                for sentence in sentences:
                    # 2. Offload ONNX inference to thread pool
                    chunks = await loop.run_in_executor(
                        executor, generate_chunks_blocking, sentence, voice, speed, lang
                    )
                    
                    # 3. Stream binary chunks immediately over socket
                    for chunk in chunks:
                        await websocket.send(chunk)

                await websocket.send(json.dumps({"type": "end", "status": "EOF", "sample_rate": 24000}))

            except Exception as e:
                logging.error(f"Error during synthesis stream: {e}")
                await websocket.send(json.dumps({"type": "error", "error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    port = 6007
    async with websockets.serve(tts_handler, "0.0.0.0", port):
        logging.info(f"🚀 Optimized Kokoro TTS running on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())