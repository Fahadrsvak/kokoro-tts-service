import asyncio
import json
import time
import numpy as np
import websockets
from kokoro_onnx import Kokoro
from misaki import en, espeak

MODEL_PATH = "kokoro-v1.0.int8.onnx"
VOICES_PATH = "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"

print("Loading Kokoro TTS model...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

print("Loading Misaki G2P (initialized once, reused across every request)...")
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)
print("Kokoro + Misaki ready.")

async def handle_tts_request(websocket):
    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue

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

            # Phonemize via Misaki instead of kokoro-onnx's default phonemizer
            # path — Misaki is dictionary-based and only shells out to espeak
            # for unknown words, avoiding the per-call backend re-init that
            # was the actual source of the multi-second delay.
            phonemes, _ = g2p(text)
            samples, sample_rate = kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True)

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