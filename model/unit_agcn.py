# The based unit of graph convolutional networks.

from model.net import conv_init
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


'''
This class implements Adaptive Graph Convolution. 
Function adapted from "Two-Stream Adaptive Graph Convolutional Networks for Skeleton Action Recognition" of Shi. et al. ("https://github.com/lshiwjx/2s-AGCN")

'''


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


class unit_agcn(nn.Module):
    def __init__(self, in_channels, out_channels,  coff_embedding=4, num_subset=3, use_local_bn=False,
                 mask_learning=False):
        super(unit_agcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.ones(6, 22, 22) * 1e-6)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)

        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, A):
        N, C, T, V = x.size()
        PA = nn.Parameter(A)
        nn.init.constant_(PA, 1e-6)
        A = A.cuda(x.get_device())
        A = Variable(A, requires_grad=False)
        A = A + PA

        y = None

        z_t = []
        for t in range(T):
            x_t = x[:, :, t, :]
            A_t = A[:, t, :, :]

            A1 = self.conv_a[0](x_t.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
            A2 = self.conv_b[0](x_t.unsqueeze(-1)).squeeze(-1)  # [N, inter_c, V]
            A_dynamic = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # [N, V, V]
            A_combined = A_t + A_dynamic
            x_t_reshaped = x_t.permute(0, 2, 1)  # [N, V, C]
            z = torch.matmul(A_combined, x_t_reshaped).permute(0, 2, 1).unsqueeze(-1)  # [N, C, V, 1]
            z = self.conv_d[0](z).squeeze(-1)  # [N, out_channels, V]
            z_t.append(z.unsqueeze(2))

        z_t = torch.cat(z_t, dim=2)  # [N, out_channels, T, V]

        y = z_t

        return self.relu(y)
