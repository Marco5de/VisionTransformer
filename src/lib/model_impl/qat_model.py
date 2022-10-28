"""
QAT in Pytorch based on blog - https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/
"""
import torch
import torch.nn as nn


class QATModel(nn.Module):

    def __init__(self, fp_model: nn.Module):
        super().__init__()
        self.fp_model = fp_model
        # QuantStub converts tensors from floating point to quantized. This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point. This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, t: torch.Tensor):
        return self.dequant(self.fp_model(self.quant(t)))
