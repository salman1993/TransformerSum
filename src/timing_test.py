import torch
import time
from extractive import ExtractiveSummarizer


fpath = "./lightning_logs/version_12/checkpoints/epoch=12.ckpt"
model = ExtractiveSummarizer.load_from_checkpoint(fpath)
model.to( torch.device("cuda:0") )

print(f"device: {model.device}")

t = "Do you like games? I want to get a refund for a order I placed. 2 items were damaged. Are you a bot? 123082HSF."

print(f"text: {t}\n\n")

k = 100
start = time.time()
for _ in range(k): summary_sents = model.predict(t, raw_scores=True)
end = time.time()

print(f"time taken: {(end-start)/k * 1000}ms")

print(summary_sents)

print("\n")

summary = model.predict(t)

print(summary)
