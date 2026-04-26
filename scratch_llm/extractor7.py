with open("data/dataset/base_training.txt","w",encoding="utf-8") as out:
    with open("data/dataset/warmup.txt","r",encoding="utf-8") as f:
        out.write(f.read())
        out.write("\n\n")
    with open("data/dataset/survival.txt","r",encoding="utf-8") as f:
        out.write(f.read())
print("Stage 2b dataset ready!")