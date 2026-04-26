import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/survivorlm.model")

text = "How do I start a fire without matches?"
tokens = sp.encode(text, out_type=str)

print(tokens)
print("Token count:", len(tokens))
