import torch
import torch.nn.functional as F


class IterativeDistiller:
    def __init__(
        self,
        teacher_model,
        student_class,
        device="cuda",
        T=2.0,
        alpha=0.5,
        num_generations=3,
        student_lr=2e-5,
        epochs_per_student=1,
    ):
        """
        teacher_model: 初代教师模型
        student_class: 用于实例化学生模型的类（每代都会重新 new 一个）
        device: "cuda" 或 "cpu"
        T: 温度系数
        alpha: soft loss 权重
        num_generations: 蒸馏的总代数
        student_lr: 学生学习率
        epochs_per_student: 每代学生训练 epoch 数
        """

        self.teacher = teacher_model.to(device)
        self.student_class = student_class
        self.device = device

        self.T = T
        self.alpha = alpha
        self.num_generations = num_generations
        self.student_lr = student_lr
        self.epochs_per_student = epochs_per_student

    # -------------------------------------------------
    # 1. 单代学生训练
    # -------------------------------------------------
    def train_one_student(self, student, dataloader):
        teacher = self.teacher
        teacher.eval()
        student.train()

        optimizer = torch.optim.Adam(student.parameters(), lr=self.student_lr)

        for epoch in range(self.epochs_per_student):
            total_loss = 0.0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # -------- 教师前向 --------
                with torch.no_grad():
                    teacher_logits = teacher(input_ids, attention_mask)
                    teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)

                # -------- 学生前向 --------
                student_logits = student(input_ids, attention_mask)
                student_probs = F.log_softmax(student_logits / self.T, dim=-1)

                # -------- soft loss (KL) --------
                loss_soft = F.kl_div(
                    student_probs, teacher_probs, reduction="batchmean"
                ) * (self.T * self.T)

                # -------- hard loss (CE) --------
                loss_hard = F.cross_entropy(student_logits, labels)

                # -------- 总蒸馏损失 --------
                loss = self.alpha * loss_soft + (1 - self.alpha) * loss_hard

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{self.epochs_per_student}  Loss: {avg_loss:.4f}")

        return student

    # -------------------------------------------------
    # 2. 多代蒸馏主函数
    # -------------------------------------------------
    def distill(self, dataloader):
        teacher = self.teacher

        for gen in range(self.num_generations):
            print(f"\n=== Generation {gen+1}/{self.num_generations} ===")

            # new 一个学生模型
            student = self.student_class().to(self.device)

            # 训练这一代学生
            student = self.train_one_student(student, dataloader)

            # 学生成为下一代教师
            teacher = student
            self.teacher = teacher

        return teacher  # 返回最后一个学生（最终模型）
