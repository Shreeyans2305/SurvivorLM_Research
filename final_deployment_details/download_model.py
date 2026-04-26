from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id="Shreyy2305/survival-gguf",
    filename="survival-q4_k_m.gguf",
    token="YOUR_HF_TOKEN_HERE",
    local_dir="/app/models",
)

size_gb = os.path.getsize(path) / (1024**3)
print(f"Downloaded: {path} ({size_gb:.2f} GB)")

if size_gb < 1.0:
    raise RuntimeError(f"File too small ({size_gb:.2f} GB) — download incomplete")

print("✅ Model file verified")