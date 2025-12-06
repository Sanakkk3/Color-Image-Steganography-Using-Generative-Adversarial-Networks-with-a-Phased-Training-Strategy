import numpy as np
from srm_filter_kernel import all_normalized_hpf_list

import torch
import torch.nn as nn

import cv2


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
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize[0], ksize[0]), sigma[k], theta, sigma[k] / 0.56, 0.5, phi[j],
                                          ktype=cv2.CV_32F)
                # print(1.5*kern.sum())
                # kern /= 1.5*kern.sum()
                filters.append(kern)
    return filters


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        filt_list = build_filters()
        # filt_list = np.array([build_filters(),build_filters(),build_filters()])

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(62, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 62, kernel_size=5, padding=2, bias=False)
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
    def __init__(self, inchannel, outchannel):
        super(block2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=2),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)

        return out
        
        
class block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, groups=32, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                #nn.ReLU(),
                #nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)
        return out
    

class UcNet(nn.Module):
  def __init__(self):
    super(UcNet, self).__init__()
    self.relu = nn.ReLU()
    self.group1 = HPF()

    self.group2 = nn.Sequential(          
      nn.Conv2d(186, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      block2(32,32)
    )
    self.group3 = block1(32,64)

    self.avgPool = nn.AvgPool2d(kernel_size=64, stride=1)
    #self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)
    self.fc1 = nn.Linear(1 * 1 * 64, 2)

  def forward(self, input):
    output = input
    
    output_y = output[:, 0, :, :]
    output_u = output[:, 1, :, :] 
    output_v = output[:, 2, :, :] 
    out_y = output_y.unsqueeze(1)
    out_u = output_u.unsqueeze(1)
    out_v = output_v.unsqueeze(1)
    y = self.group1(out_y)
    u = self.group1(out_u)
    v = self.group1(out_v)
    output = torch.cat([y, u, v], dim=1)
    #output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)

    output = self.avgPool(output)
    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output

