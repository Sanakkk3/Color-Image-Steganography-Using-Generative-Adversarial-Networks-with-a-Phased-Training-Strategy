import numpy as np
# import pandas as pd
from pathlib import Path
import scipy.io as sio
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from fixed_filter import all_filter_list
from srm_kernel import all_normalized_hpf_list

class Quant(nn.Module):
    def __init__(self, quanti):
        super(Quant, self).__init__()

        self.quanti = quanti

    def forward(self, input):
        output = input / self.quanti

        return output


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


def build_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list:
        row_1 = int((5 - hpf_item.shape[0]) / 2)
        row_2 = int((5 - hpf_item.shape[0]) - row_1)
        col_1 = int((5 - hpf_item.shape[1]) / 2)
        col_2 = int((5 - hpf_item.shape[1]) - col_1)
        hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
        filters.append(hpf_item)
    return filters


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        filt_list = build_filters()
        # filt_list = np.array([build_filters(),build_filters(),build_filters()])

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # self.quant = Quant(1.0)
        self.tlu = TLU(2.0)

        # self.sc_bn_1 = nn.BatchNorm2d(30)

        # nn.init.constant_(self.sc_bn.weight, 1.0)

    def forward(self, input):
        output = self.hpf(input)
        # output = self.quant(output)
        output = self.tlu(output)

        return output
class block2(nn.Module):
    def __init__(self):
        super(block2, self).__init__()
        self.relu = nn.ReLU()
        
        self.cov = nn.Conv2d(60, 72, kernel_size=5, stride=2, padding=2)        # 改输出90为60
        self.basic = nn.Sequential(
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, padding=1, stride=2)
        )

    def forward(self, x):
        out = torch.abs(self.cov(x))
        out = self.basic(out)
        return out


class WISENet(nn.Module):
    def __init__(self):
        super(WISENet, self).__init__()
        self.relu = nn.ReLU()
        self.group1 = HPF()
        #self.group2 = nn.Sequential(
            #nn.Conv2d(90, 72, kernel_size=5, stride=2, padding=2),
            #nn.BatchNorm2d(72),
            #nn.ReLU(),
            #nn.AvgPool2d(kernel_size=5, padding=1, stride=2)
        #)
        self.group2 = block2()
        
        self.group3 = nn.Sequential(
            nn.Conv2d(72, 288, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, padding=1, stride=4),
            
            nn.Conv2d(288, 1152, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1152),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=16, stride=1)
        )
        
        self.classfier = nn.Sequential(
            nn.Linear(1 * 1 * 1152, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 2)
        )

    def forward(self, input):
        output = input

        output_y = output[:, 0, :, :]
        output_u = output[:, 1, :, :]
        # output_v = output[:, 2, :, :]
        out_y = output_y.unsqueeze(1)
        out_u = output_u.unsqueeze(1)
        # out_v = output_v.unsqueeze(1)
        y = self.group1(out_y)
        u = self.group1(out_u)
        # v = self.group1(out_v)
        output = torch.cat([y, u], dim=1)
        output = self.group2(output)
        output = self.group3(output)
        
        output = output.view(output.size(0), -1)
        output = self.classfier(output)

        return output
