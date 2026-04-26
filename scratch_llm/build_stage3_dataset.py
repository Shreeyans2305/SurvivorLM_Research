with open("data/dataset/combined_stage3.txt","w",encoding="utf-8") as out:
    with open("data/dataset/combined_stage2.txt","r",encoding="utf-8") as f:
        out.write(f.read())
        out.write("\n\n")
    with open("data/dataset/instructions_cleaned.txt","r",encoding="utf-8") as f:
        out.write(f.read())