import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.file_names = self.get_filenames(path)
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = Image.open(self.file_names[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_filenames(self, data_path):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind("jpg") != -1 or name.rfind("png") != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        print(filename)
                        images.append(filename)
        return images
