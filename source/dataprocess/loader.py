import numpy as np
from init import datasetSorter
import skimage.io as io
import torch
from torchvision import datasets, transforms

class TrainTestLoader(torch.utils.data.Dataset):
    def __init__(self, type):
        self.dataset = datasetSorter(type)

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, index):
        # get data
        data_path, label_path = self.dataset.__getitem__(0)
        data = io.imread(data_path) / 255.0
        label = io.imread(label_path) / 33.0
        # data = self.transform(data_tensor)
        return data, label
