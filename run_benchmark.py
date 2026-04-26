import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from mamba2 import Mamba2Config
from mamba2_mc import Mamba2MCLMHeadModel


print("RUNNING NEW FILE")
# =========================
# 1 
# =========================
ckpt_path = "mamba2-ckpts/checkpoints/mamba2-finetune/Mamba2MC-final"

# =========================
# 2 tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b"
)

# =========================
# 3 config
# =========================
config = Mamba2Config(
    d_model=2048,
    n_layer=48,
    vocab_size=50288   # ??tokenizer.vocab_size
)
# =========================
# 4 load MC model
# =========================
model = Mamba2MCLMHeadModel(config)

state_dict = torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu")
result = model.load_state_dict(state_dict, strict=False)
print(f"Checkpoint has {len(state_dict)} keys")
print(f"First 5 ckpt keys: {list(state_dict.keys())[:5]}")
print(f"Missing ({len(result.missing_keys)}): {result.missing_keys[:10]}")
print(f"Unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("?MC Model loaded")

# =====================================================
#  WikiTextPerplexity?
# =====================================================

print("\n Running WikiText...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

text = "\n\n".join(dataset["text"])
enc = tokenizer(text, return_tensors="pt")

input_ids = enc["input_ids"][0]

stride = 512
max_length = 1024

nlls = []

for i in tqdm(range(0, len(input_ids) - max_length, stride)):
    chunk = input_ids[i:i+max_length].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(chunk)

    shift_logits = logits[:, :-1]
    shift_labels = chunk[:, 1:]

    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean"
    )

    nlls.append(loss)

ppl = torch.exp(torch.stack(nlls).mean())
print(" WikiText PPL:", ppl.item())

# =====================================================
#  PIQAAccuracy?
# =====================================================

print("\n Running PIQA...")

dataset = load_dataset("piqa", split="validation", trust_remote_code=True)

def score(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        logits, _ = model(inputs["input_ids"])

    return logits.mean().item()

correct = 0

for example in tqdm(dataset):
    context = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]

    s1 = score(context + " " + sol1)
    s2 = score(context + " " + sol2)

    pred = 0 if s1 > s2 else 1

    if pred == example["label"]:
        correct += 1

acc = correct / len(dataset)
print(" PIQA Accuracy:", acc)

if __name__ == "__main__":
    print(" Start running benchmark...")

    # ?
