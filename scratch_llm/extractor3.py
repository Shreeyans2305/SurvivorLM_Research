import re
import unicodedata

INPUT_FILE = "data/dataset/warmup_cleaned.txt"
OUTPUT_FILE = "data/dataset/warmup.txt"


def clean(text):
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove Unicode replacement characters
    text = text.replace('\ufffd', '')

    # Remove standalone page numbers (lines with only digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove lone figure/table labels
    text = re.sub(r'\b(figure|fig\.|table)\s+\d+\b', '', text, flags=re.IGNORECASE)

    # Remove numeric citation brackets like [1], [3, 7]
    text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)

    # Normalize bullet symbols
    text = re.sub(r'[•◦]', '-', text)

    # Fix hyphenated line breaks
    text = re.sub(r'-\n', '', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'["\']', '', text)

    return text.strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned_text = clean(raw_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaned file saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
