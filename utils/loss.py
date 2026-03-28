import torch
import torch.nn as nn

class PricingLoss(nn.Module):
    """无套利约束条件损失函数（等价最大化夏普比率）"""

    def forward(self, w: torch.Tensor, R: torch.Tensor):
        """
        Args:
            w (torch.Tensor): 条件均值方差最优组合权重Ntx1 (w)
            R (torch.Tensor): 未来收益向量Ntx1
        """
        return (1 - w.T @ R) ** 2