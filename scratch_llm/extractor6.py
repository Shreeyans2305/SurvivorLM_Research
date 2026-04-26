import re

INPUT = "data/dataset/instructions.txt"
OUTPUT = "data/dataset/instructions_cleaned.txt"

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Remove separator lines
text = re.sub(r'={5,}', '', text)

# Remove all-caps headers (simple heuristic)
text = re.sub(r'^[A-Z\s\d(),\-]{10,}$', '', text, flags=re.MULTILINE)

# Normalize spacing
text = re.sub(r'\n{3,}', '\n\n', text)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(text.strip())

print("Cleaned instructions saved.")
