import os
import random
import torch
import torch.utils.data
import cv2

# todo - class maps always identical! Sinnvoller aufteilen
_floor_class_map = {
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

_sign_class_map = {
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
    "StartlineParking": 33,
    "StopLineSigns": 34,
    "Uphill": 35,
    "XingSigns": 36
}


class SignDataset(torch.utils.data.Dataset):
    _floor_prefix = "Trainingsdaten/grey"
    _sign_prefix = "Trainingsdaten"
    _ignore_list = [".keep"]
    _implemented_sets = ["Bodenerkennung", "Schilder"]

    def __init__(self, root_dir: str, train: bool, transform=None, uniform: bool = False):
        """
        Args:
            root_dir:       root directory of dataset - path to "Bodenerkennung" directory
            train:          training mode if True else test mode - todo: test data not split
            transform:      callable applied to loaded cv image
            uniform:        uniform sampling of classes - todo not implemented
        """
        selected_set = os.path.split(root_dir)[-1]
        assert selected_set in self._implemented_sets
        if selected_set == "Bodenerkennung":
            self.class_map = _floor_class_map
            self.train_prefix = self._floor_prefix
        else:
            self.class_map = _sign_class_map
            self.train_prefix = self._sign_prefix

        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.uniform = uniform

        self.class_path_dic = {}
        for key in self.class_map.keys():
            self.class_path_dic[key] = []

        if not train:
            raise NotImplementedError("Test data is missing!")

        data_dirs = os.listdir(os.path.join(root_dir, self.train_prefix))
        for direc in data_dirs:
            if not os.path.isdir(os.path.join(root_dir, self.train_prefix, direc)):
                print(f"{direc} is not a directory - continue")
                continue

            class_dirs = os.listdir(os.path.join(root_dir, self.train_prefix, direc))
            for class_dir in class_dirs:
                if not os.path.isdir(os.path.join(root_dir, self.train_prefix, direc, class_dir)):
                    continue
                files = os.listdir(os.path.join(root_dir, self.train_prefix, direc, class_dir))
                for file in files:
                    if file in self._ignore_list:
                        continue
                    self.class_path_dic[class_dir].append(
                        os.path.join(root_dir, self.train_prefix, direc, class_dir, file))

        if uniform:
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
            selected_class = random.randrange(start=0, stop=self.class_map["XingSigns"])
            raise NotImplementedError()
        else:
            img_path = self.path_list[item]
            # extract class from path
            label = self.class_map[img_path.split(os.sep)[-2]]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, label
