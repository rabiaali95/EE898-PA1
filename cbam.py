import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ChannelBranch(nn.Module):
    def __init__(self, in_channel, r =16 , pool_types = ['avg, max']):
        super(ChannelBranch, self).__init__()
        self.pool_types = pool_types
        self.mlp = mlp(in_channel,r)

    def forward(self,x):
        # getting size of the input
        batch, channel, H, W = x.size()
        #Average and Max pooling is done to aggragate the spatial information of feature map
        for pooling in self.pool_types:
            if pooling == 'avg':
                avg_pool = F.avg_pool2d(x,(H,W),stride=(H,W)).squeeze(3).squeeze(2) #Averaged pooled feature map


            elif pooling == 'max':
                max_pool = F.max_pool2d(x,(H,W),stride=(H,W)).squeeze(3).squeeze(2) #Max pooled feature map
        #Both maps are separately passed to MLP with 1 hidden layer
        mlp1 = self.mlp(avg_pool)
        mlp2 = self.mlp(max_pool)
        #Produces combined channel attention map
        mlp_sum = mlp1+mlp2
        channel_att = F.sigmoid(mlp_sum) #This is inaccordance with eq 2 of CBAM paper
        channel_att = channel_att.unsqueeze(2).unsqueeze(3).expand_as(x)

        return x*channel_att

#MLP with one hidden layer is used to estimate attention across channels from the channel vector.
class mlp(nn.Module):
    def __init__(self, in_channel, r=16):
        super(mlp, self).__init__()
        self.mlp = nn.Sequential(
            #Hidden activation size is set to R^(C/r x 1 x 1)
            nn.Linear(in_channel, in_channel // r),
            nn.ReLU(),
            #Hidden activation size is set back to R^(C x 1 x 1)
            nn.Linear(in_channel // r, in_channel)
        )
    def forward(self,x):
        mlp_out = self.mlp(x)
        return mlp_out

class SpatialBranch(nn.Module):
    def __init__(self, in_plane=2, out_plane=1,):
        super(SpatialBranch, self).__init__()
        self.spatial = nn.Sequential(
            #Concatenated averaged and max pooling feature map is convolved by convolution layer to produce 2D spatial attention map (R^(HxW)
            nn.Conv2d(in_plane, out_plane, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_plane,eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU()
        )
    def forward(self,x):

        #Average pooling and max pooling is done along channel axis. They are then concatenated to form an efficient feature descriptor
        ch_pool = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_att = self.spatial(ch_pool)
        spatial_att = F.sigmoid(spatial_att) #In accordance with eq 3 of the CBAM paper
        return x*spatial_att

class CBAM(nn.Module):
    def __init__(self, in_channel, r=16, pool_types=['avg', 'max'], bool_spatial=False):
        super(CBAM, self).__init__()
        self.ch = ChannelBranch(in_channel, r, pool_types)
        #self.bool_spatial = bool_spatial
        if bool_spatial:
            self.spatial = SpatialBranch()
    def forward(self, x):
        # Channel wise attention is applied first
        out = self.ch(x)
        # Spatial attention is only applied if the bool is true
        if self.bool_spatial:
            out = self.spatial(out)
        return out






