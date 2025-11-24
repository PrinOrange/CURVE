import torch
import torch.nn.functional as F
from KD import IterativeDistiller
from CweBert.Classifier import RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 假设有 tokenized 输入
input_ids = torch.randint(0, 1000, (32, 128))
attention_mask = torch.ones_like(input_ids)
labels = torch.randint(0, 2, (32,))

dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=8)

# 初代教师
teacher_model = RobertaForSequenceClassification().to(DEVICE)

distiller = IterativeDistiller(
    teacher_model=teacher_model,
    student_class=RobertaForSequenceClassification,
    device=DEVICE,
    num_generations=3,
    student_lr=2e-5,
    T=2.0,
    alpha=0.5,
    epochs_per_student=1,
)

final_student = distiller.distill(dataloader)
