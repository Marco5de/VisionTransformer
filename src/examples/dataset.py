"""
Example showing how to work with the torch dataset for signs and floor signs
"""
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms

from tqdm import tqdm
from src.lib.dataset.SignDataset import SignDataset


def test_floor(dataset_dir: str, stats: bool = False):
    """
    Create dataset and loader, use essential transforms, plot images, optionally compute statistical properties (for normalization)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1221), (0.2415))  # mean and var
    ])
    dset = SignDataset(dataset_dir, train=True, transform=transform, uniform=False)
    print("Length of dataset: ", len(dset))
    loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True)
    img, label = next(iter(loader))

    fig, axarr = plt.subplots(1, 4)
    for idx, ax in enumerate(axarr):
        ax.imshow(img[idx].squeeze(), cmap="gray")
        ax.set_axis_off()
    fig.show()

    if stats:
        # compute mean of dataset
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)
        mean = torch.zeros([1], dtype=torch.float32)
        std = torch.zeros([1], dtype=torch.float32)
        for img, label in tqdm(loader):
            mean += img.mean()
            std += img.std()

        print(f"Mean of dataset - {mean / len(loader)} Std of dataset - {std / len(loader)}")


def test_sign(dataset_dir):
    """
    Create dataset and loader, apply essential transforms, plot example images
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dset = SignDataset(dataset_dir, train=True, transform=transform, uniform=False)
    print(f"Length of dataset = {len(dset)}")

    loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True)
    img, label = next(iter(loader))

    fig, axarr = plt.subplots(1, 4)
    for idx, ax in enumerate(axarr):
        ax.imshow(img[idx].permute([1, 2, 0]).flip(2), cmap="gray")
        ax.set_axis_off()
    fig.show()


def __main__():
    test_floor("data/Bodenerkennung", stats=False)
    test_sign("data/Schilder")


if __name__ == "__main__":
    __main__()
