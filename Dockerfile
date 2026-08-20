FROM python:3.11-slim

WORKDIR /app

# Install system audio libraries (espeak-ng for phonemizer, libsndfile1 for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY download_weights.py .
RUN python download_weights.py

COPY server.py .

EXPOSE 6007

CMD ["python", "server.py"]