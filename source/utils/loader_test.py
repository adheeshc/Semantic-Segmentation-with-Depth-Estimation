import os
import numpy as np
from skimage.transform import warp, AffineTransform

import torch
import torch.utils.data as data
import torchvision.utils
from torchvision import transforms

class NYU_Depth_V2(data.Dataset):
    def __init__(self, train = True):
        if train == True:
            num_images = np.int(500 * 0.9)
        else:
            num_images = np.int(500 * 0.1)
        self.images = np.random.rand(num_images, 3,  228, 304) * 256
        self.depth = np.random.rand(num_images, 1, 480, 640)
        self.transform = transforms.Compose([
            transforms.Resize(size=(240,320), interpolation=1),
            transforms.CenterCrop(size=(228,304))
            ])

        return

    def __len__(self):
        if hasattr(self, 'images'):
            length = len(self.images)
            print("Length", length)

        return length

    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depth[index]

        # The below call takes care of the image transformation. But one of the need is to get the data in the compatible format. The numpy array is not transformed using this logic
        #image_transformed = self.transform(image)

        return image, depth
