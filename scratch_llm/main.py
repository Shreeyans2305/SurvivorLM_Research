import sentencepiece as spm
import os

INPUT_FILE = "data/tokenizer/tokenizer_corpus.txt"
MODEL_PREFIX = "data/tokenizer/survivorlm"

os.makedirs("data/tokenizer", exist_ok=True)

spm.SentencePieceTrainer.Train(
    input=INPUT_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    shuffle_input_sentence=True,
)

print("Tokenizer training complete.")
