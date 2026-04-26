import os

DATASET = [
    "data/dataset/warmup.txt",
    "data/dataset/survival.txt",
    "data/dataset/instructions.txt"
]

OUT_FILE = "data/tokenizer/tokenizer_corpus.txt"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE,"w",encoding="utf-8") as out:
    for path in DATASET:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            out.write(f.read())
            out.write("\n\n")
print("Tokenizer Corpus Created!")