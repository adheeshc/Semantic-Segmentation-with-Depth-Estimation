import numpy as np
import os
import skimage.io as io
from init import datasetSorter
from loader import TrainTestLoader
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def unit_test_loss():
    """
    Function to unit test the reverse hubber loss class
    :return:
    """
    x = torch.randn(2, 2, requires_grad=True)
    target = torch.randn(2, 2, requires_grad=True)
    # print(x)
    # print(target)

    loss_fn = processor.ReverseHubberLoss()
    output = loss_fn.forward(x, target)
    print(output)
    print(output.backward())
    return output

def weight_data(m):
    """
    Unit test to apply a function to print weight data
    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d):
        print(m.weight.data)

def unit_test_model():
    """
    Function to unit test the model
    :return:
    """
    model = classifier.DepthPredictionNet()
    print(model)
    print(x.shape)
    y = model.forward(x)
    print(y.shape)
    return

type = 'train'
# dataset = datasetSorter(type)
# print(dataset.__getitem__(2))
train_set = TrainTestLoader(type)
img, lab = train_set.__getitem__(1)
img = img.numpy().squeeze().astype('float32')
img_ = np.zeros((img.shape[1], img.shape[2], 3))
img_[:,:,0] = img[0,:,:]
img_[:,:,1] = img[1,:,:]
img_[:,:,2] = img[2,:,:]
img__ = img_.astype('float32')
print(img__.shape)
io.imshow(img__)
io.show()
#
lab = lab.numpy()#.astype('float32')
# img_[:,:,0] = lab[0,:,:]
# img_[:,:,1] = lab[0,:,:]
# img_[:,:,2] = lab[0,:,:]
# img__ = img_.astype('float32')
# img__ = img__/np.amax(img__)
# io.imshow(img__)
# io.show()

img_ = np.zeros((img.shape[1], img.shape[2], 3))
img_[:,:,0] = lab[0,:,:]#*((output>0) & (output <= 0.33)).astype(int)
img_[:,:,1] = lab[0,:,:]#*((output>0.33) & (output <= 0.66)).astype(int)
img_[:,:,2] = lab[0,:,:]#*((output>0.66)).astype(int)
img_ = rgb2gray(img_)
img__ = img_.astype('float32')
plt.imshow(img__)
plt.show()
