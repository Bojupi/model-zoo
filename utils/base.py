import torch
import torch.nn as nn

class Base(nn.Module):
    """基类"""

    def count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self):
        base = super().__repr__()
        params = self.count()
        return f"{base}\nParameters Count: {params:,}"
    
    def export(self, path: str, input_shape: tuple):
        """将模型导出为onnx格式
        Args: 
            path (str): 模型保存路径（需以`.onnx`为文件名后缀）
            input_shape (tuple): 输入特征维度
        """
        self.eval()
        backup = next(self.parameters()).device
        self.to("cpu")
        input_tensor = torch.randn(input_shape, dtype=torch.float32)
        torch.onnx.export(
            self, 
            input_tensor,
            path,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        self.to(backup)