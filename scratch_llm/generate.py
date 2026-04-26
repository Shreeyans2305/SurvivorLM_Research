import torch
import sentencepiece as spm
from model.gpt import SurvivorLM
from model.config import SurvivorLMConfig
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/survivorlm.model")

config = SurvivorLMConfig()
model = SurvivorLM(config).to(DEVICE)
model.load_state_dict(torch.load("survivorlm.pt",map_location=DEVICE))
model.eval()

def generate(prompt,max_tokens=120,temperature=0.6,top_k=20):
    tokens = sp.encode(prompt,out_type=int)
    tokens = torch.tensor(tokens,dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_tokens):
        tokens_cond = tokens[:,-config.max_seq_len:]
        with torch.no_grad():
            logits = model(tokens_cond)
        logits = logits[:,-1,:]/temperature
        for token_id in set(tokens[0].tolist()):
            logits[0, token_id] /= 1.2
        if top_k is not None:
            v, _ = torch.topk(logits,top_k)
            logits[logits<v[:,[-1]]] = -float("inf")

        probs = torch.softmax(logits,dim=1)
        next_token = torch.multinomial(probs,num_samples=1)
        tokens = torch.cat([tokens,next_token],dim=1)

    output = sp.decode(tokens[0].tolist())
    return output

if __name__ == "__main__":
    prompt = f"Q: How do I purify water in the wilderness?\nA:"
    print(generate(prompt))