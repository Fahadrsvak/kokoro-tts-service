# Kokoro TTS Service

A high-performance, low-latency Text-to-Speech (TTS) WebSocket streaming microservice powered by the **Kokoro-82M ONNX** model. Designed for real-time voice agents, turn-based assistants, and edge deployments via Docker & Coolify.

## Features

- **Streaming Synthesis**: Streams PCM audio chunks progressively as text is processed for low Time-To-First-Byte (TTFB).
- **Lightweight & Efficient**: ONNX runtime execution with ~350MB RAM footprint on CPU.
- **WebSocket Protocol**: Native full-duplex socket streaming returning 16-bit 24kHz raw PCM binary audio frames.
- **Coolify Ready**: Pre-configured `Dockerfile` and `docker-compose.yml` for zero-downtime deployment on shared internal Docker networks (`coolify`).
- **Pre-baked Weights**: ONNX weights and voice binaries are downloaded during image build for rapid container boot times.

---

## Tech Stack

- **Model**: Kokoro-82M ONNX (v1.0)
- **Engine**: `kokoro-onnx` + `onnxruntime`
- **Networking**: `websockets` (Python `asyncio`)
- **Phonemization**: `espeak-ng`
- **Containerization**: Docker / Docker Compose

---

## Project Structure

```text
kokoro-tts-service/
├── Dockerfile           # Multi-stage image build with espeak-ng & weight download
├── docker-compose.yml   # Coolify / local Docker Compose setup
├── requirements.txt     # Python dependencies
├── download_weights.py  # Build-time script to fetch Kokoro ONNX model and voices
└── server.py            # Async WebSocket server handling streaming TTS payloads