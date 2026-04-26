import os

CLEAN_DIR = "more/warmup"
OUT_FILE = "data/dataset/warmup_cleaned.txt"

os.makedirs("data/dataset", exist_ok=True)

with open(OUT_FILE,"w",encoding="utf-8") as out:
    for filename in os.listdir(CLEAN_DIR):
        if not filename.lower().endswith(".txt"):
            continue
        file_path = os.path.join(CLEAN_DIR, filename)
        print("Processing:", file_path)
        with open(file_path,"r",encoding="utf-8") as f:
            out.write("\n\n")
            out.write(f.read())
print("Saved combined dataset to:", OUT_FILE)