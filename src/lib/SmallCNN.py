import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):

    def __init__(self, num_classes: int, dropout_rate: float, input_channels: int=1):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        self.linear1 = nn.Linear(2816, 256, bias=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, t: torch.Tensor):
        t = self.feature_extractor(t)
        _, c, h, w = t.shape
        t = t.reshape(shape=(-1, c * h * w))
        t = F.relu(self.dropout1(self.linear1(t)))
        t = F.softmax(self.linear2(t), dim=-1)
        return t


if __name__ == "__main__":
    from src.lib.utils import count_parameters

    t = torch.randn(size=(8, 1, 160, 40))
    cnn = SmallCNN(num_classes=37, dropout_rate=0.1)
    out = cnn(t)

    print(f"Number of parameters = {count_parameters(cnn)}")
