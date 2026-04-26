import re

INPUT = "data/dataset/instructions_upsampled.txt"
OUTPUT = "data/dataset/instructions_strict.txt"

with open(INPUT, "r", encoding="utf-8") as f:
    text = f.read()

# Extract only Q/A blocks
qa_blocks = re.findall(r'Q:.*?A:.*?(?=\nQ:|\Z)', text, re.DOTALL)

cleaned = "\n\n".join(block.strip() for block in qa_blocks)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(cleaned)

print("Strict Q/A file created:", OUTPUT)
