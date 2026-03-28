# 论文标题: Artificial Intelligence Asset Pricing Models
# 引用: Kelly B T, Kuznetsov B, Malamud S, et al. Artificial intelligence asset pricing models[R]. National Bureau of Economic Research, 2025.
# doi链接: https://doi.org/10.3386/w33351

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from utils.base import Base


class Block(nn.Module):
    """AIPM中的Encoder块"""

    def __init__(self, input_size: int, num_heads: int, hidden_size: int, dropout: float):
        """
        Args:
            input_size (int): 输入特征维度 (D)
            num_heads (int): 注意力头数 (H)
            hidden_size (int): 隐藏层维度 (df)
            dropout (float): 元素置零的概率
        """
        super().__init__()

        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.W = nn.ParameterList(
            [nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/input_size), size=(input_size, input_size))) for _ in range(num_heads)]
        )
        self.V = nn.ParameterList(
            [nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/input_size), size=(input_size, input_size))) for _ in range(num_heads)]
        )

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(1/hidden_size))
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(hidden_size, input_size, bias=True)
        nn.init.normal_(self.fc2.weight, mean=0, std=np.sqrt(1/input_size))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Block输入，形状为NxD (Y)
        Returns:
            torch.Tensor: Block输出，形状为NxD
        """
        out = torch.zeros_like(X)

        # 多头注意力 + 残差连接 (first sublayer)
        for h in range(self.num_heads):
            W_h = self.W[h]
            V_h = self.V[h]
            A_h = F.softmax(X @ W_h @ X.T, dim=1) @ X @ V_h
            out += A_h
        out = self.dropout(out) + X

        # 全连接前馈+残差连接 (second sublayer)
        out = self.fc2(F.relu(self.fc1(out))) + out

        return out


class AIPM(Base):
    """Artificial Intelligence Asset Pricing Model 人工智能资产定价模型"""

    def __init__(self, input_size: int, num_heads: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Args:
            input_size (int): 输入特征维度 (D)
            num_heads (int): 注意力头数 (H)
            hidden_size (int): 隐藏层维度 (df)
            num_layers (int): 层块数 (K)
            dropout (float): 元素置零的概率
        """
        super().__init__()

        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
            [Block(input_size, num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )

        self.lmbda = nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/input_size), size=(input_size, 1)))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): 股票截面特征NtxD (X)
        Returns:
            torch.Tensor: 条件均值方差最优组合权重 (w)
        """
        for k in range(self.num_layers):
            block = self.blocks[k]
            X = block(X)
        
        # 将w标准化为多空权重
        w = X @ self.lmbda
        w = w - w.mean()
        w = w / w.abs().sum()
        return w
    
    def fit(self, train_dset: list[torch.Tensor], valid_dset: list[torch.Tensor], lr: float, epochs: int, weight_decay: float, criterion: callable):
        """参数训练
        Args:
            train_dset (list[torch.Tensor]): 股票特征训练数据集，每个元素均为Ntx(D+1)矩阵，前D列为特征，最后一列为未来收益
            train_dset (list[torch.Tensor]): 股票特征验证数据集，每个元素均为Ntx(D+1)矩阵，前D列为特征，最后一列为未来收益
            lr (float): 学习率
            epochs (int): 训练epoch
            weight_decay (float): 正则项系数
            criterion (callable): 损失函数
        """

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # 早停配置
        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for sample in tqdm(train_dset):
                optimizer.zero_grad()
                X = sample[:, :-1]
                R = sample[:, -1:]
                loss = criterion(self(X), R)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_loss /= len(train_dset)
            val_loss = self.valid(valid_dset, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_count += 1
            print(f"Epoch{str(epoch+1).zfill(2)} | Train: {epoch_loss:.10f} | Val: {val_loss:.10f}")
            if patience_count >= 5:
                print(f"Early Stopping at Epoch {epoch+1}")
                break
        if best_state is not None:
            self.load_state_dict(best_state)
    
    def valid(self, valid_dset: list[torch.Tensor], criterion: callable):
        """验证
        Args:
            train_dset (list[torch.Tensor]): 股票特征验证数据集，每个元素均为Ntx(D+1)矩阵，前D列为特征，最后一列为未来收益
            criterion (callable): 损失函数
        """
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for sample in valid_dset:
                X = sample[:, :-1]
                R = sample[:, -1:]
                loss = criterion(self(X), R)
                val_loss += loss.item()
        val_loss /= len(valid_dset)
        return val_loss


if __name__ == "__main__":

    from utils.loss import PricingLoss

    D = 30
    T = 100
    N = torch.randint(3000, 5000, size=(100, ))
    X = [torch.normal(0, 1, (N[t], D)) for t in range(T)]
    R = [torch.normal(0, 0.01, (N[t], 1)) for t in range(T)]
    XR = [torch.cat([X[t], R[t]], dim=1) for t in range(T)]

    m = AIPM(X[0].shape[1], 2, 64, 2, 0.3)
    m.fit(XR, XR, 1e-4, 10, 1e-3, PricingLoss())