import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class TextDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,seq_len=512):
        self.seq_len = seq_len

        with open(text_path,"r",encoding="utf-8",errors="ignore") as f:
            text = f.read()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load (tokenizer_path)
        tokens = self.sp.encode(text,out_type=int)

        self.data = []
        for i in range(0,len(tokens)-seq_len,seq_len):
            chunk = tokens[i:i+seq_len+1]
            self.data.append(chunk)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1],dtype=torch.long)
        y = torch.tensor(chunk[1:],dtype=torch.long)
        return x,y