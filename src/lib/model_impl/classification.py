import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNet(nn.Module):
    # todo - use this to replace simple CNN - attaches classification head to any feature extractor

    def __init__(self, backbone: nn.Module, num_classes, dropout_rate=0.1):
        super().__init__()
        self.backbone = backbone

        self.linear1 = nn.Linear(2816, 256, bias=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, t: torch.Tensor):
        t = self.backbone(t)
        print(t.shape)
        _, c, h, w = t.shape
        t = t.reshape(shape=(-1, c * h * w))

        t = F.relu(self.dropout1(self.linear1(t)))
        t = F.softmax(self.linear2(t), dim=-1)
        return t
