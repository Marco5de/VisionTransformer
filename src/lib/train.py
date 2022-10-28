from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter


def train_signs_epoch(model: nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      optim,
                      epoch: int,
                      device,
                      criterion,
                      writer: SummaryWriter):
    # set model to training mode (e.g. activate dropout)
    model.train()
    train_acc = 0
    for batch_idx, (input, label) in tqdm(enumerate(data_loader), total=len(data_loader)):
        input, label = input.to(device).float(), label.to(device)
        optim.zero_grad()

        pred = model(input)
        loss = criterion(pred, label)
        loss.backward()
        optim.step()

        cls = pred.argmax(dim=1, keepdim=True)
        train_acc += cls.eq(label.view_as(cls)).sum().item()

        writer.add_scalar("train/loss", loss.item(), epoch * len(data_loader) + batch_idx)

    writer.add_scalar("train/epoch_acc", train_acc / len(data_loader.dataset), epoch)
