from datasets import load_dataset

# 1. Enable streaming=True to avoid the massive download
dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

output_file = "natural.txt"
target_size = 17 * 1024 * 1024  # 17 MB in bytes
current_size = 0

# 2. Iterate and stop once the limit is hit
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        text_to_write = entry["text"] + "\n\n"
        encoded_text = text_to_write.encode("utf-8")
        
        # Check if adding this entry exceeds 17MB
        if current_size + len(encoded_text) > target_size:
            # Optional: write the remaining portion to hit exactly 17MB
            remaining = target_size - current_size
            f.write(encoded_text[:remaining].decode("utf-8", errors="ignore"))
            break
            
        f.write(text_to_write)
        current_size += len(encoded_text)

print(f"Done! Created {output_file} (Size: {current_size / (1024*1024):.2f} MB)")
