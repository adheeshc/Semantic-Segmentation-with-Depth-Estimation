import os
import sys
import numpy as np
from utils import processor1, loader
from net import classifier1
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch
import torchlight
import skimage.io as io
from torchsummary import summary

device = 'cuda:0'
model = classifier1.DepthPredictionNet().to(device)
print('Model loaded')
summary(model, (3, 228, 304))
