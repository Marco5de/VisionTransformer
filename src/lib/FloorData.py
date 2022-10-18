import os
import torch
import torch.utils.data

__class_map = {
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

    def __init__(self, root_dir: str, train: bool):
        self.root_dir = root_dir
        self.train = train
        if not train:
            raise NotImplementedError("Test data is missing!")

        data_dirs = os.listdir(os.path.join(root_dir, self._train_prefix))
        for direc in data_dirs:
            if not os.path.isdir(os.path.join(root_dir, self._train_prefix, direc)):
                print("Not a dir continue", direc)
                continue

        print(data_dirs)



    def __len__(self):
        pass

    def __getitem__(self, item: int):
        pass



def __main__():
    dset = FloorDataset("data/", train=True)


if __name__ == "__main__":
    __main__()