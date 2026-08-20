FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (espeak-ng is required by Kokoro phonemizer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ONNX model and voices into container image
COPY download_weights.py .
RUN python download_weights.py

COPY server.py .

EXPOSE 8880

CMD ["python", "server.py"]