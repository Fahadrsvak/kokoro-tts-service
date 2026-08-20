import asyncio
import json
import time
import numpy as np
import websockets
from kokoro_onnx import Kokoro
from misaki import en, espeak

MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"
DEFAULT_VOICE = "af_heart"

print("Loading Kokoro TTS model...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

print("Loading Misaki G2P...")
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

            await websocket.send(json.dumps({"type": "start", "text": text}))

            g2p_start = time.time()
            phonemes, _ = g2p(text)
            g2p_ms = int((time.time() - g2p_start) * 1000)

            synth_start = time.time()
            samples, sample_rate = kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
            synth_ms = int((time.time() - synth_start) * 1000)

            print(f"[TIMING] text={text!r} g2p_ms={g2p_ms} synth_ms={synth_ms}", flush=True)

            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            await websocket.send(pcm16.tobytes())

            await websocket.send(json.dumps({
                "type": "end",
                "text": text,
                "sample_rate": sample_rate,
                "g2p_ms": g2p_ms,
                "synth_ms": synth_ms,
                "synthesis_ms": g2p_ms + synth_ms
            }))

    except Exception as e:
        print(f"Connection error: {e}", flush=True)

async def main():
    async with websockets.serve(handle_tts_request, "0.0.0.0", 6007, max_size=None):
        print("Kokoro TTS WebSocket server listening on port 6007...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())