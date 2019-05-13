from torch import nn
class SELayer(nn.Module):
    def __init__(self, in_channel, r=16):
        super(SELayer, self).__init__()
        # Adaptive Average Pooling Layer- this refers to the Squeeze block in the paper
        self.squeeze = Pooling()

        # This captures channel-wise dependencies. This refers to Excitation block in the paper.
        # This implementation exactly follows equation 3 in the SE attention paper.
        self.excitation = nn.Sequential()
        self.excitation.add_module('sq_ex_fc1', nn.Linear(in_channel, in_channel//r))
        self.excitation.add_module('sq_ex_relu', nn.ReLU(inplace=True))
        self.excitation.add_module('sq_ex_fc2', nn.Linear(in_channel//r, in_channel))
        self.excitation.add_module('sq_ex_sigmoid', nn.Sigmoid())

    def forward(self, x):

        # applying the squeeze operation
        squeeze = self.squeeze(x)
        #  applying the excitation operation
        excitation = self.excitation(squeeze)
        sq_ex_scale = excitation.unsqueeze(2).unsqueeze(3)
        sq_ex_scale_expanded = sq_ex_scale.expand_as(x)
        return x*sq_ex_scale_expanded

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