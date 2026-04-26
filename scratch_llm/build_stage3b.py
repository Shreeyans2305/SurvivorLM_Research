with open("data/dataset/combined_stage3b.txt","w",encoding="utf-8") as out:
    with open("data/dataset/survival.txt","r",encoding="utf-8",errors="ignore") as f:
        out.write(f.read())
        out.write("\n\n")
    with open("data/dataset/instructions_upsampled.txt","r",encoding="utf-8",errors="ignore") as f:
        out.write(f.read())
        out.write("\n\n")
print("Stage 3b dataset ready!")