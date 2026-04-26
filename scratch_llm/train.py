import torch
from torch.utils.data import DataLoader
from model.gpt import SurvivorLM
from model.config import SurvivorLMConfig
from scratch_llm.dataset import TextDataset
from model.instruction_dataset import InstructionDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

config = SurvivorLMConfig()
model = SurvivorLM(config).to(DEVICE)
model.load_state_dict(torch.load("survivorlm.pt", map_location=DEVICE))

dataset = InstructionDataset(
    text_path="data/dataset/instructions_strict.txt",
    tokenizer_path="data/tokenizer/survivorlm.model",
    seq_len = config.max_seq_len,
)

loader = DataLoader(dataset,batch_size=8,shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()

for epoch in range(1):
    total_loss = 0.0
    for x,y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits.reshape(-1, config.vocab_size),y.reshape(-1))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}| Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "survivorlm.pt")
print("Training complete and model saved as survivorlm.pt")