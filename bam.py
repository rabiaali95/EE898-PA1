import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ChannelBranch(nn.Module):
    def __init__(self, in_channel, r=16):
        super(ChannelBranch, self).__init__()
        self.ch = nn.Sequential(
            #Adaptive average pooling is done to produce channel vector that belongs to R^(Cx1x1)
            Pooling(),
            #MLP with one hidden layer is used to estimate attention across channels from the channel vector. Hidden activation size is set to R^(C/r x 1 x 1)
            nn.Linear(in_channel, in_channel // r),
            #Batch Normalization is added to adjust scale with spatial branch output
            nn.BatchNorm1d(in_channel // r),
            nn.ReLU(),
            # Hidden activation size is set back to R^(C x 1 x 1)
            nn.Linear(in_channel // r, in_channel)
        )

    def forward(self, x):
        ch = self.ch(x) #Steps are in accordance with eq 3 of BAM paper
        ch_scale = ch.unsqueeze(2).unsqueeze(3)
        ch_scale_expanded = ch_scale.expand_as(x)
        return ch_scale_expanded

class SpatialBranch(nn.Module):
    def __init__(self, in_channel, r=16, dilation_val=4):
        super(SpatialBranch, self).__init__()

        self.spatial = nn.Sequential(
            #Initially feature map has dimention R^(C x H x W). 1x1 convolution is done to project this feature map to reduced dimension R^(C/r x H x W)
            nn.Conv2d(in_channel, in_channel // r, kernel_size=1),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            #Dilation convolution is done to enlarge the receptive fields with high efficiency
            #Dilation Convolution is done twice

            #First
            nn.Conv2d(in_channel // r, in_channel // r, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            #Second
            nn.Conv2d(in_channel // r, in_channel // r, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            #Fetures are further reduced to R^(1 x H x W) using 1x1 convolution
            nn.Conv2d(in_channel // r, 1, kernel_size=1)
        )

    def forward(self, x):

        output = self.spatial(x)  #Steps are in accordance with eq 4 of BAM paper
        output = output.expand_as(x)
        return output

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # getting size of the input
        batch, channel, _, _ = x.size()
        pooled = self.pooling(x)
        pooled = pooled.squeeze(3).squeeze(2)
        return pooled


class BAM(nn.Module):
    def __init__(self, in_channel, r=16):
        super(BAM, self).__init__()
        self.channel_attention = ChannelBranch(in_channel, r)
        self.spatial_attention = SpatialBranch(in_channel, r)

    def forward(self, x):
        #Spatial and Channel attention branches are computed seperately and then either added or multiplied
        #This eq refers to eq 1 of paper where F i.e input feature map (x in my case) is taken as common and multiplied in next line
        full_attention = 1 + F.sigmoid(self.channel_attention(x) + self.spatial_attention(x))

        return x * full_attention
