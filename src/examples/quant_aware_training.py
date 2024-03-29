"""
QAT in Pytorch based on blog - https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/
"""

import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18

import os
import datetime
import copy

from src.lib.dataset.SignDataset import SignDataset
from src.lib.model_impl.qat_model import QATModel
from src.lib.train import train_signs_epoch
from src.lib.utils.utils import export_onnx

transform = transforms.Compose([
    transforms.ToTensor(),
])
dset = SignDataset("data/Schilder", train=True, transform=transform, uniform=False)
# dset = SignDataset("data/Bodenerkennung", train=True, transform=transform, uniform=False)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

# todo - resnet subsampling is too much for such small images!
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 37)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Use {device} for training")
model.to(device)
model.float()

criterion = nn.CrossEntropyLoss()
now = datetime.datetime.now()
model_path = f"model/model_{now.strftime('%d-%m-%y-%H-%M-%S')}.onnx"
writer = SummaryWriter(log_dir=os.path.join("log_dir", f"model_{now.strftime('%d-%m-%y-%H-%M-%S')}"))

for epoch in range(2):
    train_signs_epoch(model, train_loader, optim, epoch, device, criterion, writer)
    export_onnx(model, device, (1, 3, 80, 80) , model_path)
model = model.cpu()
fused_model = copy.deepcopy(model)

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
quant_model.to(device)

now = datetime.datetime.now()
quant_model_path = f"../model/quant_model_{now.strftime('%d-%m-%y-%H-%M-%S')}.pth"
quant_writer = SummaryWriter(log_dir=os.path.join("log_dir", quant_model_path))

for epoch in range(3):
    train_signs_epoch(quant_model, train_loader, optim, epoch, device, criterion, quant_writer)
    # todo - quantized models are saved differently - see blog
    # torch.save(quant_model.state_dict(), quant_model_path)

quant_model = quant_model.cpu()
quant_model = torch.quantization.convert(quant_model, inplace=True)
quant_model.eval()