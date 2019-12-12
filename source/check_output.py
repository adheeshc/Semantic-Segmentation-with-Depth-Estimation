import os
import sys
import numpy as np
from utils import processor
from dataprocess import loader
from net import classifier
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch
import torchlight
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def model_load(path_file, model):
    if path_file == True:
        print('Please enter a valid path')
        return 0
    path_file = './model_output/' + path_file
    checkpoint = torch.load(path_file)
    model.load_state_dict(checkpoint)
    return model

model = classifier.DepthPredictionNet()
path_file = 'epoch1_acc1.00_model.pth.tar'
model = model_load(path_file, model)
device = 'cuda:0'
model.to(device)
a = loader.TrainTestLoader(True)
d,l = a.__getitem__(0)

#
# print(type(e))
# print(type(l))
# print(l.shape)
# print(d.shape)
# print(np.amax(d))
# print(np.amax(l))
img_ = np.zeros((l.shape[1], l.shape[2], 3))
l = l.detach().cpu().clone().numpy()

e = d.detach().cpu().clone().numpy()

img_[:,:,0] = e[0,:,:]
img_[:,:,1] = e[1,:,:]
img_[:,:,2] = e[2,:,:]
img__ = img_.astype('float32')
io.imsave('input.png', img__)
io.imshow(img__)
io.show()


img_[:,:,0] = l[0,:,:]
img_[:,:,1] = l[0,:,:]
img_[:,:,2] = l[0,:,:]
img_ = rgb2gray(img_)
img__ = img_.astype('float32')
plt.imshow(img__)
plt.savefig('ground_truth.png')
plt.show()

# io.imsave('ground_truth.png', img__)
# io.imshow(img__, cmap=)
# io.show()

d.unsqueeze_(0)
d = d.to(device)
print(type(d))
output = model(d)
print(type(output))
print(output.shape)
output = output.detach().cpu().clone().numpy()
output = output[0,:,:,:].copy()
# print(np.amax(d))
# print(np.amax(l))
img_ = np.zeros((output.shape[1], output.shape[2], 3))
img_[:,:,0] = output[0,:,:]#*((output>0) & (output <= 0.33)).astype(int)
img_[:,:,1] = output[0,:,:]#*((output>0.33) & (output <= 0.66)).astype(int)
img_[:,:,2] = output[0,:,:]#*((output>0.66)).astype(int)
img_ = rgb2gray(img_)
img__ = img_.astype('float32')
plt.imshow(img__)
plt.savefig('output.png')
plt.show()
# img__ = img_.astype('float32')
# io.imsave('output.png', img__)
# img__ = img_.astype('float32')
# io.imshow(i
