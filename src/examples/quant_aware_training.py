"""
QAT in Pytorch based on blog - https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/
"""
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18

import os
import datetime
import copy

from src.lib.SignDataset import SignDataset
from src.lib.train import train_signs_epoch
from src.lib.model_impl.qat_model import QATModel
from src.lib.utils import inference_time


def __main__():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dset = SignDataset("data/Schilder", train=True, transform=transform, uniform=False)
    dset = SignDataset("data/Bodenerkennung", train=True, transform=transform, uniform=False)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

    # todo - custom model erstellen / es fehlt noch ein classification head!
    model = resnet18(pretrained=True, )
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Use {device} for training")
    model.to(device)
    model.float()

    criterion = nn.CrossEntropyLoss()
    now = datetime.datetime.now()
    model_path = f"../model/model_{now.strftime('%d-%m-%y-%H-%M-%S')}.pth"
    writer = SummaryWriter(log_dir=os.path.join("log_dir", model_path))

    # todo test model benchmark
    print("Running inference latency test")
    ret = inference_time(model, device, input_size=[1, 3, 160, 40], num_warmup=10, num_runs=100)
    print(f"Average inference time = {ret * 1e-6}ms")

    """
    # todo - train model non quantized
    for epoch in range(10):
        train_signs_epoch(model, train_loader, optim, epoch, device, criterion, writer)
        torch.save(model.state_dict(), model_path)
    """
    model = model.cpu()
    fused_model = copy.deepcopy(model)

    # todo fusion stuff did not work as shown in the blog - research why and how it is used!
    # model.train()
    # fused_model.train()
    # todo according to blog train should be used - triggers pytorch assertion!
    model.eval()
    fused_model.eval()

    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]],
                                                inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    quant_model = QATModel(model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quant_model.qconfig = quantization_config
    print(quant_model.qconfig)

    torch.quantization.prepare_qat(quant_model, inplace=True)
    quant_model.train()

    now = datetime.datetime.now()
    quant_model_path = f"../model/quant_model_{now.strftime('%d-%m-%y-%H-%M-%S')}.pth"
    quant_writer = SummaryWriter(log_dir=os.path.join("log_dir", quant_model_path))
    """
    # todo - train model quantized
    for epoch in range(10):
        train_signs_epoch(quant_model, train_loader, optim, epoch, device, criterion, quant_writer)
        torch.save(quant_model.state_dict(), quant_model_path)
    """
    quant_model = quant_model.cpu()
    quant_model = torch.quantization.convert(quant_model, inplace=True)
    quant_model.eval()


if __name__ == "__main__":
    __main__()
