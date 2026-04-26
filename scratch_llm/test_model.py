import torch
from model.gpt import SurvivorLM
from model.config import SurvivorLMConfig

config = SurvivorLMConfig()
model = SurvivorLM(config)

dummy_input = torch.randint(0,config.vocab_size,(2,16))
logits = model(dummy_input)

print("Logits shape: ",logits.shape)