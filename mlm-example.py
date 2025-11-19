import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn.functional as F

# This program can test MLM pre-training effect with actual code snippets
# The following `code_to_test` provides an example, using <mask> to mask code tokens
# Predict the masked tokens and output corresponding confidence scores

# -----------------------------
# Input example, input C/C++ code,
# You can input your test code here.
# Note: use <mask> to replace the masked tokens
# -----------------------------
CODE_TO_TEST = """
FILE* openFile(<mask> flag) {
    FILE* f = fopen("data.txt", "r");
    if (flag) return nullptr;
    return f;
}
"""

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "/home/malaoshi/bak/checkpoint-20000"
TOP_K_TO_PREDICT = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# Load model and tokenizer
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

masked_text = CODE_TO_TEST.replace("<mask>", tokenizer.mask_token)
inputs = tokenizer(masked_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(DEVICE)

# -----------------------------
# Find mask position
# -----------------------------
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    logits = model(input_ids).logits

# Get logits at mask position
mask_logits = logits[0, mask_token_index, :][0]

# Apply softmax to get probabilities
probs = F.softmax(mask_logits, dim=-1)

# Get top-k predictions
top_probs, top_indices = torch.topk(probs, TOP_K_TO_PREDICT)

print("Top predictions for <mask>:")
for token_id, prob in zip(top_indices, top_probs):
    token_str = tokenizer.decode([token_id])
    print(f"{token_str:20s}  prob={prob.item():.6f}")

# -----------------------------
# Construct replaced code
# -----------------------------
best_token = tokenizer.decode([top_indices[0]])
predicted_code = masked_text.replace(tokenizer.mask_token, best_token)

print("\nPredicted most probably Code:\n", predicted_code)