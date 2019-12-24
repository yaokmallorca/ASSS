import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DisSigmoidFCN(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """
    def __init__(self,in_channels,negative_slope = 0.2):
        super(DisSigmoidFCN, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.norm1 = nn.GroupNorm(8,64)
        # self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.norm2 = nn.GroupNorm(8,128)
        # self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.norm3 = nn.GroupNorm(8,64)
        # self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        return x

class DisSigmoid(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """
    def __init__(self,in_channels,negative_slope = 0.2):
        super(DisSigmoid, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=2,dilation=1)
        self.norm1 = nn.GroupNorm(8,64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=2,dilation=1)
        self.norm2 = nn.GroupNorm(8,128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=2,dilation=1)
        self.norm3 = nn.GroupNorm(8,256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=2,dilation=1)
        self.norm4 = nn.GroupNorm(8,512)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=2,padding=2,dilation=1)

    def forward(self,x):
        x = self.conv1(x) # -,-,320
        x = self.norm1(x)
        x = self.relu1(x)
        # print(x.size())
        x = self.conv2(x) # -,-,160
        x = self.norm2(x)
        x = self.relu2(x)
        # print(x.size())
        x = self.conv3(x) # -,-,80
        x = self.norm3(x)
        x = self.relu3(x)
        # print(x.size())
        x = self.conv4(x) # -,-,40
        x = self.norm4(x)
        x = self.relu4(x)
        # print(x.size())
        x = self.conv5(x) # -,-,20
        # print(x.size())
        # upsample
        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-, 31,31
        # print(x.size())

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-,41,41
        # print(x.size())

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,81,81
        # print(x.size())

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,161,161
        # print(x.size())

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-2,:-2] # -,-,321,321
        # print(x.size())

        return x



