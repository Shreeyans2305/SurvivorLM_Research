from pdfminer.high_level import extract_text
import os
import re
RAW_DIR = "more/survival"
OUT_DIR = "data/extracted2"

os.makedirs(OUT_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(RAW_DIR, filename)
    print("Extracting text from:", pdf_path)
    text = extract_text(pdf_path)

    text = re.sub(r'\b(figure|fig\.|table)\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}','\n\n',text)
    text = re.sub(r'[ \t]+',' ',text)
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)
    out_name = os.path.splitext(filename)[0] + ".txt"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved extracted text to:", out_path)