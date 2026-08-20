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
MODEL_PATH = os.path.join(WEIGHTS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(WEIGHTS_DIR, "voices-v1.0.bin")

logging.info(f"Loading Standard FP32 ONNX Model from: {MODEL_PATH}")

# ---------------------------------------------------------
# CPU THREADING & GRAPH OPTIMIZATIONS
# ---------------------------------------------------------
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Force ONNX Runtime to use all available physical CPU threads
cpu_cores = os.cpu_count() or 4
session_options.intra_op_num_threads = cpu_cores
session_options.inter_op_num_threads = 2

# Check for CUDA GPU provider vs CPU fallback
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]

session = ort.InferenceSession(MODEL_PATH, session_options=session_options, providers=providers)
logging.info(f"Active Execution Providers: {session.get_providers()}")
logging.info(f"Allocated Intra-Op CPU Threads: {cpu_cores}")

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

            try:
                for sentence_idx, sentence in enumerate(sentences):
                    logging.info(f"\n==================================================")
                    logging.info(f"Processing [{sentence_idx+1}/{len(sentences)}]: '{sentence}'")
                    
                    # ---------------------------------------------------------
                    # STAGE 1: G2P / Phonemizer Timing
                    # ---------------------------------------------------------
                    g2p_start = time.perf_counter()
                    phonemes = kokoro.tokenizer.phonemize(sentence, lang)
                    g2p_duration = (time.perf_counter() - g2p_start) * 1000
                    
                    logging.info(f"🗣️ [STAGE 1] G2P / Phonemizer (espeak-ng): {g2p_duration:.2f} ms")
                    logging.info(f"   ↳ Phonemes: '{phonemes}'")

                    # ---------------------------------------------------------
                    # STAGE 2: ONNX Model Inference & Streaming
                    # ---------------------------------------------------------
                    stream_start = time.perf_counter()
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
                            logging.info(f"⚡ [STAGE 2] First Audio Frame ONNX Inference: {first_chunk_onnx:.2f} ms")
                            logging.info(f"🚀 [TTFA] Total Latency To First Audio Frame: {ttfa_ms:.2f} ms")
                            first_chunk = False

                        # STAGE 3: PCM Quantization Timing
                        pcm_start = time.perf_counter()
                        pcm_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                        pcm_bytes = pcm_samples.tobytes()
                        total_pcm_time += (time.perf_counter() - pcm_start) * 1000

                        # STAGE 4: WebSocket Network Send Timing
                        send_start = time.perf_counter()
                        await websocket.send(pcm_bytes)
                        total_send_time += (time.perf_counter() - send_start) * 1000

                    sentence_stream_duration = (time.perf_counter() - stream_start) * 1000
                    
                    logging.info(f"🎛️ [STAGE 2] Total ONNX Stream Synthesis: {sentence_stream_duration:.2f} ms")
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
        logging.info(f"🚀 Optimized Standard Kokoro TTS running on ws://0.0.0.0:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())