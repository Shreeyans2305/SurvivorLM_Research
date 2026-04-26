FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ cmake make git wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Build llama-cpp-python from source — ensures GGUF format compatibility
# This takes ~10 minutes but supports all GGUF versions
RUN pip install llama-cpp-python --no-cache-dir

RUN pip install fastapi uvicorn huggingface_hub --no-cache-dir

COPY download_model.py .
RUN python download_model.py

COPY main.py .

ENV MODEL_PATH="/app/models/survival-q4_k_m.gguf"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]