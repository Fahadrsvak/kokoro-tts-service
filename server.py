import asyncio
import json
import os
import re
import time
import logging
import websockets
import numpy as np
from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

logging.info("Initializing Kokoro ONNX model into memory...")
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

async def warmup():
    try:
        logging.info("Warming up execution pipeline...")
        async for _ in kokoro.create_stream("Warmup test.", voice="af_heart", speed=1.0, lang="en-us"):
            pass
        logging.info("Warmup complete.")
    except Exception as err:
        logging.warning(f"Warmup warning: {err}")

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

async def tts_handler(websocket):
    try:
        async for message in websocket:
            request_start_time = time.perf_counter()
            
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

            await websocket.send(json.dumps({"type": "start", "sample_rate": 24000}))

            sentences = split_into_sentences(text)

            try:
                for sentence_idx, sentence in enumerate(sentences):
                    logging.info(f"\n==================================================")
                    logging.info(f"Processing [{sentence_idx+1}/{len(sentences)}]: '{sentence}'")
                    
                    # ---------------------------------------------------------
                    # 1. ISOLATED G2P / PHONEMIZATION STEP (espeak-ng)
                    # ---------------------------------------------------------
                    g2p_start = time.perf_counter()
                    # Access internal tokenizer directly
                    phonemes = kokoro.tokenizer.phonemize(sentence, lang)
                    g2p_duration = (time.perf_counter() - g2p_start) * 1000
                    
                    logging.info(f"🗣️ [STAGE 1] G2P / Phonemizer (espeak-ng): {g2p_duration:.2f} ms")
                    logging.info(f"   ↳ Phonemes generated: '{phonemes}'")

                    # ---------------------------------------------------------
                    # 2. ISOLATED ONNX INFERENCE + STREAMING STEP
                    # ---------------------------------------------------------
                    stream_start = time.perf_counter()
                    # Pass is_phonemes=True to skip duplicate phonemization
                    stream = kokoro.create_stream(
                        text=phonemes,
                        voice=voice,
                        speed=speed,
                        lang=lang,
                        is_phonemes=True
                    )

                    chunk_count = 0
                    first_chunk = True
                    total_pcm_time = 0.0
                    total_send_time = 0.0

                    async for samples, sample_rate in stream:
                        chunk_count += 1
                        
                        if first_chunk:
                            ttfa_ms = (time.perf_counter() - request_start_time) * 1000
                            first_chunk_onnx = (time.perf_counter() - stream_start) * 1000
                            logging.info(f"⚡ [STAGE 2] First Audio Frame ONNX Synthesis: {first_chunk_onnx:.2f} ms")
                            logging.info(f"🚀 [TTFA] Total Latency To First Audio Frame: {ttfa_ms:.2f} ms")
                            first_chunk = False

                        # Measure PCM quantization timing
                        pcm_start = time.perf_counter()
                        pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                        pcm_bytes = pcm_samples.tobytes()
                        total_pcm_time += (time.perf_counter() - pcm_start) * 1000

                        # Measure WebSocket send timing
                        send_start = time.perf_counter()
                        await websocket.send(pcm_bytes)
                        total_send_time += (time.perf_counter() - send_start) * 1000

                    sentence_stream_duration = (time.perf_counter() - stream_start) * 1000
                    
                    logging.info(f"🎛️ [STAGE 2] Total ONNX Synthesis Streaming: {sentence_stream_duration:.2f} ms")
                    logging.info(f"   ↳ PCM Array Processing Total: {total_pcm_time:.2f} ms")
                    logging.info(f"   ↳ Network Send Total: {total_send_time:.2f} ms")
                    logging.info(f"✅ Sentence Total: {(time.perf_counter() - g2p_start) * 1000:.2f} ms | Chunks: {chunk_count}")

                await websocket.send(json.dumps({"type": "end", "status": "EOF", "sample_rate": 24000}))
                
                total_request_time = (time.perf_counter() - request_start_time) * 1000
                logging.info(f"🏁 End-to-End Request Duration: {total_request_time:.2f} ms\n")

            except Exception as e:
                logging.error(f"Error during synthesis stream: {e}", exc_info=True)
                await websocket.send(json.dumps({"type": "error", "error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    port = 6007
    await warmup()
    async with websockets.serve(tts_handler, "0.0.0.0", port):
        logging.info(f"🚀 Profiling Kokoro TTS running on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())