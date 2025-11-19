import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn.functional as F

# -----------------------------
# 输入示例，输入 C/C++ 代码，
# 注意用 <mask> 替换掉被遮挡的词
# -----------------------------
code_to_test = """
struct Data {
    int x;
};

void foo(std::Function<void()> f) {
    f();
}

int main() {
    Data* p = new Data();
    auto f = <mask>::bind([](Data* d) { d->x = 42; }, p);
    foo(f);
    return 0;
}
"""

# -----------------------------
# 配置
# -----------------------------
MODEL_DIR = "/home/malaoshi/bak/checkpoint-14000"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# 加载模型与 tokenizer
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForMaskedLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

masked_text = code_to_test.replace("<mask>", tokenizer.mask_token)
inputs = tokenizer(masked_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(DEVICE)

# -----------------------------
# 找 mask 的位置
# -----------------------------
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# -----------------------------
# 推理
# -----------------------------
with torch.no_grad():
    logits = model(input_ids).logits

# mask 位置的 logits
mask_logits = logits[0, mask_token_index, :][0]

# softmax 得概率
probs = F.softmax(mask_logits, dim=-1)

# top-k
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

print("Top predictions for <mask>:")
for token_id, prob in zip(top_indices, top_probs):
    token_str = tokenizer.decode([token_id])
    print(f"{token_str:20s}  prob={prob.item():.6f}")

# -----------------------------
# 构造替换后的代码
# -----------------------------
best_token = tokenizer.decode([top_indices[0]])
predicted_code = masked_text.replace(tokenizer.mask_token, best_token)

print("\nPredicted most probably Code:\n", predicted_code)
