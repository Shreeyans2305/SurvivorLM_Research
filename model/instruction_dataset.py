import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class InstructionDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,seq_len):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        with open(text_path,"r",encoding="utf-8") as f:
            text = f.read()
        raw_blocks = text.split("Q:")
        self.examples = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            block = "Q:" + block
            if "A:" not in block:
                continue
            q_part, a_part = block.split("A:",1)
            q_text = q_part.strip() + " A:"
            a_text = a_part.strip()

            q_tokens = self.sp.encode(q_text,out_type=int)
            a_tokens = self.sp.encode(" " + a_text, out_type=int)

            input_tokens = q_tokens + a_tokens
            target_tokens = [-100] * len(q_tokens) + a_tokens

            if len(input_tokens) > seq_len:
                input_tokens = input_tokens[:seq_len]
                target_tokens = target_tokens[:seq_len]

            pad_length = seq_len - len(input_tokens)
            if pad_length > 0:
                input_tokens += [0]*pad_length
                target_tokens += [-100]*pad_length
            self.examples.append((
                torch.tensor(input_tokens,dtype=torch.long),
                torch.tensor(target_tokens,dtype=torch.long)
            ))

    def __len__(self):
        return len(self.examples)
    def __getitem__(self,idx):
        return self.examples[idx]