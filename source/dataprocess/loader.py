import numpy as np
from dataprocess.init import datasetSorter
import skimage.io as io
import torch
from torchvision import datasets, transforms

class TrainTestLoader(torch.utils.data.Dataset):
    def __init__(self, type):
        self.dataset = datasetSorter(type)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.ToPILImage(),
                                             transforms.Resize(size=(240, 320), interpolation=1),
                                              transforms.CenterCrop(size=(228, 304)), transforms.ToTensor()])

    def __len__(self):
        return self.dataset.len()

    def __getitem__(self, index):
        # get data
        data_path, label_path = self.dataset.getitem(index)
        data = io.imread(data_path) / 255.0
        label = io.imread(label_path).astype('float32')
        label[label > 0] = (label[label > 0] - 1) / 256
        data = data.astype('float32')
        label = label.astype('float32')/np.amax(label)
        data = self.transform(data)
        label = self.transform(label)
        # data = self.transform(data_tensor)
        return data, label
