import asyncio
import json
import time
import numpy as np
import websockets
from kokoro_onnx import Kokoro

MODEL_PATH = "kokoro-v1.0.int8.onnx"
VOICES_PATH = "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"

print("Loading Kokoro TTS model...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
print("Kokoro TTS model loaded.")

async def handle_tts_request(websocket):
    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue  # this server expects JSON text requests, not binary

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            text = data.get("text", "").strip()
            voice = data.get("voice", DEFAULT_VOICE)
            speed = float(data.get("speed", 1.0))

            if not text:
                continue

            start = time.time()
            await websocket.send(json.dumps({"type": "start", "text": text}))

            samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

            # Kokoro returns float32 samples in [-1, 1] — convert to int16 PCM
            # so the client can play it directly via the Web Audio API.
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            await websocket.send(pcm16.tobytes())

            synthesis_ms = int((time.time() - start) * 1000)
            await websocket.send(json.dumps({
                "type": "end",
                "text": text,
                "sample_rate": sample_rate,
                "synthesis_ms": synthesis_ms
            }))

    except Exception as e:
        print(f"Connection error: {e}")

async def main():
    async with websockets.serve(handle_tts_request, "0.0.0.0", 6007, max_size=None):
        print("Kokoro TTS WebSocket server listening on port 6007...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())