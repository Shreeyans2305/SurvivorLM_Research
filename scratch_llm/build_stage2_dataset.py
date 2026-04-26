with open("data/dataset/combined_stage2.txt","w",encoding="utf-8") as out:
    with open("data/dataset/survival.txt","r",encoding="utf-8",errors="ignore") as f:
        out.write(f.read())
        out.write("\n\n")
    with open("data/dataset/warmup.txt","r",encoding="utf-8",errors="ignore") as f:
        out.write(f.read())
        out.write("\n\n")
print("Combined Stage 2 Dataset Created!")