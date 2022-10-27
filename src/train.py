import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from src.lib.SignDataset import SignDataset
from src.lib.ViT import ClsTransformer
from src.lib.SmallCNN import SmallCNN
from src.lib.utils import count_parameters




def train(model, train_loader, optim, epoch, device, criterion):
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (input, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input, label = input.to(device), label.to(device)
        optim.zero_grad()
        # todo hack for wrong labels
        pred = model(input)[:, 0, :]
        # pred = model(input)
        loss = criterion(pred, label)
        loss.backward()
        optim.step()

        train_loss += loss.sum().item()
        cls = pred.argmax(dim=1, keepdim=True)
        train_acc += cls.eq(label.view_as(cls)).sum().item()

    print(
        f"Epoch: {epoch + 1} Test-loss: {train_loss / len(train_loader.dataset)} Acc: {train_acc / len(train_loader.dataset)}")


def test(model, test_loader, device, criterion, epoch):
    model.eval()
    test_loss, acc = 0, 0
    for batch_idx, (input, label) in enumerate(test_loader):
        input, label = input.to(device), label.to(device)
        pred = model(input)[:, 0, :]
        test_loss += criterion(pred, label).sum().item()
        cls = pred.argmax(dim=1, keepdim=True)
        acc += cls.eq(label.view_as(cls)).sum().item()


def __main__():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1221), (0.2415))
    ])
    dset = SignDataset("data/", train=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

    # note, number of parameters of transformer is very sensitive to embedding dimension! (size of all linear layers)
    #model = ClsTransformer(num_classes=37, embed_dim=64, input_dim=[16, 1, 160, 40], num_heads=4, num_layers=6)
    # todo - debug why smallCNN model does not work well
    model = SmallCNN(num_classes=37, dropout_rate=0.1)
    model.double()
    print(f"#parameter = {count_parameters(model)}")

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        train(model, train_loader, optim, epoch, device, criterion)


if __name__ == "__main__":
    __main__()
