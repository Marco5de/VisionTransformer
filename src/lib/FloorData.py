import os
import random
import torch
import torch.utils.data
import cv2

_class_map = {
    "BarredArea": 0,
    "Downhill": 1,
    "ExpressEnd": 2,
    "ExpressStart": 3,
    "GiveWaySigns": 4,
    "LeftArrowSigns": 5,
    "NoClass": 6,
    "NoPassingEnd": 7,
    "NoPassingStart": 8,
    "Pedestrian": 9,
    "PedestrianIsland": 10,
    "RightArrowSigns": 11,
    "RightOfWay": 12,
    "SharpTurnLeft": 13,
    "SharpTurnRight": 14,
    "Speed10EndSigns": 15,
    "Speed10Signs": 16,
    "Speed20EndSigns": 17,
    "Speed20Signs": 18,
    "Speed30EndSigns": 19,
    "Speed30Signs": 20,
    "Speed40EndSigns": 21,
    "Speed40Signs": 22,
    "Speed50EndSigns": 23,
    "Speed50Signs": 24,
    "Speed60EndSigns": 25,
    "Speed60Signs": 26,
    "Speed70EndSigns": 27,
    "Speed70Signs": 28,
    "Speed80EndSigns": 29,
    "Speed80Signs": 30,
    "Speed90EndSigns": 31,
    "Speed90Signs": 32,
    "StartLineParking": 33,
    "StopLineSigns": 34,
    "Uphill": 35,
    "XingSigns": 36
}


class FloorDataset(torch.utils.data.Dataset):
    _train_prefix = "Bodenerkennung/Trainingsdaten/grey"
    _ignore_list = [".keep"]

    def __init__(self, root_dir: str, train: bool, transform=None, uniform: bool=False):
        """
        Args:
            root_dir:       root directory of dataset - path to "Bodenerkennung" directory
            train:          training mode if True else test mode - todo: test data not split
            transform:      callable applied to loaded cv image
            uniform:        uniform sampling of classes
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.uniform = uniform

        self.class_path_dic = {}
        for key in _class_map.keys():
            self.class_path_dic[key] = []

        if not train:
            raise NotImplementedError("Test data is missing!")

        data_dirs = os.listdir(os.path.join(root_dir, self._train_prefix))
        for direc in data_dirs:
            if not os.path.isdir(os.path.join(root_dir, self._train_prefix, direc)):
                print("Not a dir continue", direc)
                continue

            class_dirs = os.listdir(os.path.join(root_dir, self._train_prefix, direc))
            for class_dir in class_dirs:
                if not os.path.isdir(os.path.join(root_dir, self._train_prefix, direc, class_dir)):
                    continue
                files = os.listdir(os.path.join(root_dir, self._train_prefix, direc, class_dir))
                for file in files:
                    if file in self._ignore_list:
                        continue
                    self.class_path_dic[class_dir].append(
                        os.path.join(root_dir, self._train_prefix, direc, class_dir, file))

        if  uniform:
            raise NotImplementedError("Not yet implemented, required classes unclear")
        else:
            self.path_list = []
            for v in self.class_path_dic.values():
                self.path_list.extend(v)

        for k, v in self.class_path_dic.items():
            print(f"Found {len(v)} {k} images")

    def __len__(self):
        length = 0
        for (k, v) in self.class_path_dic.items():
            length += len(v)
        return length

    def __getitem__(self, item: int):
        if self.uniform:
            selected_class = random.randrange(start=0, stop=_class_map["XingSigns"])
            raise NotImplementedError()
        else:
            img_path = self.path_list[item]
            # extract class from path
            label = _class_map[img_path.split("/")[-2]]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def __main__():
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1221), (0.2415))
    ])
    dset = FloorDataset("data/", train=True, transform=transform, uniform=False)
    print("Length of dataset: ", len(dset))
    loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True)
    img, label = next(iter(loader))

    fig, axarr = plt.subplots(1, 4)
    for idx, ax in enumerate(axarr):
        ax.imshow(img[idx].squeeze(), cmap="gray")
        ax.set_axis_off()
    fig.show()

    compute_mean = False
    if compute_mean:
        # compute mean of dataset
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)
        mean = torch.zeros([1], dtype=torch.float32)
        std = torch.zeros([1], dtype=torch.float32)
        for img, label in tqdm(loader):
            mean += img.mean()
            std += img.std()

        print(f"Mean of dataset - {mean / len(loader)} Std of dataset - {std / len(loader)}")

if __name__ == "__main__":
    __main__()
