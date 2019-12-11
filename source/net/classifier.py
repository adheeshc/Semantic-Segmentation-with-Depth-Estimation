import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResnetCustom(nn.Module):
    """
    Custom resnet layer with the Fully convolution layer removed
    """
    # TBD: Weight initialization function in processor1.py, the upsample function used in our paper is different, I am
    # not able to find a function in torch for that. Either we may have to implement our own.
    def __init__(self):
        """
        Initialize the Resnet Custom layer without the Fully connected layer
        """
        super(ResnetCustom, self).__init__()
        #Loading the pretrained resnet50_model and removing the last 2 layers
        resnet50_model = models.resnet50(pretrained=True)
        self.model_features = nn.Sequential(*list(resnet50_model.children())[:-2])
        # print(self.model_features)

    def get_resnet_layers(self):
        """
        Function to return the 48 layers of Resnet-50.
        :return: Returns the list of sequential layers
        """
        return self.model_features

    def forward(self, a_input):
        a_out = self.model_features(a_input)
        return a_out

class UpconvLayer(nn.Module):
    """
    Upconvolution layer
    """
    def __init__(self, a_in_channels , a_out_channels, a_kernel_size=5, a_stride=1):
        super(UpconvLayer, self).__init__()
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(a_in_channels, a_out_channels, kernel_size=a_kernel_size, stride=a_stride, padding=2)
        self.conv.weight.data.normal_(0.0, 0.01)

    def forward(self, a_input):
        # a_hidden = self.zeropad(a_input)
        a_hidden = self.unpool(a_input)
        a_out = self.conv(a_hidden)
        return a_out

    def zeropad(self, x, stride=2):
        w = x.new_zeros(stride, stride)
        w[0, 0] = 1
        out = F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))
        #print("Shape before zero padding", x.shape)
        #print("Data before zero padding", x)
        #print("Shape after zero padding", out.shape)
        #print("Data after padding", out)
        return out

class DepthPredictionNet(nn.Module):
    """
    Depth prediction net with upconvolutions and pretrained resnet 50 model
    """
    def __init__(self):
        """
        Initialize the depth prediction layer
        """
        super(DepthPredictionNet, self).__init__()
        self.resnet_pretrained = ResnetCustom()
        print(self.resnet_pretrained)
        self.conv1 = nn.Conv2d(2048, 1024, 1)
        self.conv1.weight.data.normal_(0.0, 0.01)
        self.norm1 = nn.BatchNorm2d(1024)
        self.norm1.weight.data.normal_(0.0, 0.01)
        # Batch normalization to be added in this layer
        self.upconv1 = UpconvLayer(1024, 512)
        self.upconv2 = UpconvLayer(512, 256)
        self.upconv3 = UpconvLayer(256, 128)
        self.upconv4 = UpconvLayer(128, 64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(p= 0.5, inplace=False)
        self.conv2.weight.data.normal_(0.0, 0.01)
        self.upsample = nn.UpsamplingBilinear2d(size=(256, 512))

    def forward(self, a_input):
        """
        Forward computation graph for the Depth prediction network
        :param a_input:
        :return:
        """
        #print(a_input.shape)
        l_hidden1 = self.resnet_pretrained(a_input)
        l_hidden2 = self.norm1(self.conv1(l_hidden1))
        #print(l_hidden2.shape)
        l_hidden3 = F.relu(self.upconv1(l_hidden2))
        #print(l_hidden3.shape)
        l_hidden4 = F.relu(self.upconv2(l_hidden3))
        #print(l_hidden4.shape)
        l_hidden5 = F.relu(self.upconv3(l_hidden4))
        #print(l_hidden5.shape)
        l_hidden6 = F.relu(self.upconv4(l_hidden5))
        l_hidden7 = self.dropout1(l_hidden6)
        #print(l_hidden6.shape)
        l_output1 = F.relu(self.conv2(l_hidden7))
        #print(l_output1.shape)
        l_output = self.upsample(l_output1)
        #print(l_output.shape)
        return l_output
