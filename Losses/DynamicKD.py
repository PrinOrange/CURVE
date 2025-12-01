import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicKDLoss(nn.Module):
    """
    动态加权的知识蒸馏损失 L_KD
    L_KD = u * w * L_hard + L_soft + L_int
    """
    def __init__(self, T=2.0, num_total_cwe=944):
        super().__init__()
        self.T = T
        self.num_total_cwe = num_total_cwe
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # 不求和，便于加权
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def compute_hard_loss(self, s_logits, y_true, u_weight, w_comp):
        """ L_hard = u * w * CE(y_true, s_logits) """
        raw_ce = self.ce_loss(s_logits, y_true)
        # 加权平均
        weighted_ce = (raw_ce * u_weight * w_comp).mean()
        return weighted_ce

    def compute_soft_loss(self, t_logits, s_logits):
        """ L_soft = T^2 * KL(p_t, p_s) """
        # 使用温度平滑
        t_soft = self.softmax(t_logits / self.T)
        s_log_soft = self.log_softmax(s_logits / self.T)
        
        # KL散度: KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        # P是目标/教师，Q是预测/学生
        # KLDivLoss(log_Q, P)
        soft_loss = self.kl_div(s_log_soft, t_soft) * (self.T**2)
        return soft_loss

    def compute_int_loss(self, t_int_R, s_int_R):
        """ L_int = 1 - sim(R_t, R_s) """
        # 使用余弦相似度
        similarity = self.cosine_sim(t_int_R, s_int_R).mean()
        int_loss = 1.0 - similarity
        return int_loss
    
    def compute_u_weight(self, t_logits):
        """ 不确定性权重 u = 1 - Entropy / log(C) """
        p = self.softmax(t_logits)
        entropy = -torch.sum(p * self.log_softmax(t_logits), dim=1)
        
        # 归一化熵
        log_c = torch.log(torch.tensor(self.num_total_cwe, dtype=torch.float32)).to(t_logits.device)
        normalized_entropy = entropy / log_c
        
        u_weight = 1.0 - normalized_entropy
        return u_weight
    
    # 类别补偿系数 w 的计算 (需要外部传入 CWE 频率信息)
    def compute_w_comp(self, cwe_count, total_count, current_cwe_id, beta_param=0.999):
        """ 类别补偿系数 w """
        current_cwe_count = cwe_count.get(current_cwe_id, 1)
        # 动态 β 逻辑 (这里简化，论文中更复杂)
        # 实际 β 需根据当前 IR 动态调整
        
        # w = (1 - beta) / (1 - beta^N_k)
        if current_cwe_count == 0:
            return 1.0
            
        beta = beta_param # 假设一个固定的 β
        w_comp = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), torch.tensor(current_cwe_count)))
        return w_comp