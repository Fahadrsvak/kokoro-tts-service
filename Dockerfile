FROM python:3.11-slim
WORKDIR /app

# espeak-ng as a defensive system fallback for phonemization — kokoro-onnx's
# pip dependencies typically bundle their own binary, but this costs almost
# nothing and avoids a build-time surprise if that assumption doesn't hold.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir kokoro-onnx websockets numpy

# int8 quantized model (~80MB) — good fit for the 2-core ARM box.
# Swap to kokoro-v1.0.onnx (fp32, ~300MB) if quality matters more than
# footprint once you've benchmarked the int8 version.
RUN wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx \
    && wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY server.py /app/

EXPOSE 6007
CMD ["python", "server.py"]