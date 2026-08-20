import asyncio
import json
import os
import re
import time
import logging
import websockets
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WEIGHTS_DIR = "weights"
# Use INT8 quantized model if available for ~2-3x speedup on CPU
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.int8.onnx") 
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")

VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

logging.info(f"Loading ONNX Model: {MODEL_PATH}")

# ---------------------------------------------------------
# OPTIMIZATION 1: Tune ONNX Runtime Session Options
# ---------------------------------------------------------
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Allocate physical CPU threads
num_cores = os.cpu_count() or 4
session_options.intra_op_num_threads = num_cores
session_options.inter_op_num_threads = 2

# Check Execution Providers (CUDA vs CPU)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]

session = ort.InferenceSession(MODEL_PATH, session_options=session_options, providers=providers)
logging.info(f"Active Execution Providers: {session.get_providers()}")

kokoro = Kokoro.from_session(session, VOICES_PATH)

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
            first_global_frame = True

            for sentence_idx, sentence in enumerate(sentences):
                sentence_start = time.perf_counter()
                
                # 1. G2P Phase
                phonemes = kokoro.tokenizer.phonemize(sentence, lang)

                # 2. ONNX Generation Phase
                stream = kokoro.create_stream(
                    text=phonemes,
                    voice=voice,
                    speed=speed,
                    lang=lang,
                    is_phonemes=True
                )

                async for samples, sample_rate in stream:
                    if first_global_frame:
                        ttfa_ms = (time.perf_counter() - request_start_time) * 1000
                        logging.info(f"🚀 [TTFA] Overall Latency To First Audio Frame: {ttfa_ms:.2f} ms")
                        first_global_frame = False

                    # Convert to PCM16
                    pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                    await websocket.send(pcm_samples.tobytes())

                sentence_time = (time.perf_counter() - sentence_start) * 1000
                logging.info(f"✅ Sentence [{sentence_idx+1}/{len(sentences)}] completed in {sentence_time:.2f} ms")

            await websocket.send(json.dumps({"type": "end", "status": "EOF", "sample_rate": 24000}))
            
            total_time = (time.perf_counter() - request_start_time) * 1000
            logging.info(f"🏁 Request Completed in: {total_time:.2f} ms\n")

    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    port = 6007
    await warmup()
    async with websockets.serve(tts_handler, "0.0.0.0", port):
        logging.info(f"🚀 Optimized Kokoro TTS running on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())