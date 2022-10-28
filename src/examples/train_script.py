import datetime
import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.lib.train import train_signs_epoch
from src.lib.dataset.SignDataset import SignDataset
from src.lib.model_impl.SmallCNN import SmallCNN


def train_signs():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dset = SignDataset("data/Schilder", train=True, transform=transform, uniform=False)
    dset = SignDataset("data/Bodenerkennung", train=True, transform=transform, uniform=False)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

    # model = SmallCNN(num_classes=37, dropout_rate=.1, input_channels=1)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Use {device} for training")
    model.to(device)
    model.float()

    criterion = nn.CrossEntropyLoss()

    now = datetime.datetime.now()
    model_path = f"../model/model_{now.strftime('%d-%m-%y-%H-%M-%S')}.pth"
    writer = SummaryWriter(log_dir=os.path.join("log_dir", model_path))

    for epoch in range(10):
        train_signs_epoch(model, train_loader, optim, epoch, device, criterion, writer)
        torch.save(model.state_dict(), model_path)


def __main__():
    train_signs()


if __name__ == "__main__":
    __main__()
