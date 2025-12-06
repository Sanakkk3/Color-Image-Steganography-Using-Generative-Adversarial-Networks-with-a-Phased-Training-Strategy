import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import scipy.io as sio
import logging
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

from hpf import *
from ResUNet1DB import UNet
from covnet1 import CovNet
from wisenet1 import WISENet

TS_PARA = 60
PAYLOAD = 0.4

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='alaska',default='alaska')    # dataset
    parser.add_argument('--dataroot', help='path to dataset',default='/dataroot',help="path to traning set") 
    
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=25, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

    parser.add_argument('--niter', type=int, default=72, help='number of epochs to train for')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netWise', default='', help="path to netWise (to continue training)")
    parser.add_argument('--netCov', default='', help="path to netCov (to continue training)")
    parser.add_argument('--outf', default='asym', help='folder to output images and model checkpoints')
    
    parser.add_argument('--config', default='G7_cov_wise_DDLloss_withW_R_pub', help='config for training') # 
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    

    opt = parser.parse_args()
    return opt

# 彩色图像处理
class ALADataset256(data.Dataset):
    def __init__(self, root, transforms = None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = np.array([0, 1], dtype='int32')     

        data = cv2.imread(img_path, -1)         
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)   
        data = np.transpose(data, (2,0,1))     
        rand_ud = np.random.rand(3, 256, 256)   
        sample = {'data': data, 'rand_ud': rand_ud, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self):
        return len(self.imgs)

class ALAToTensor():
    def __call__(self, sample):
        data, rand_ud, label = sample['data'], sample['rand_ud'], sample['label']

        # 读彩色图像不需要，因为维度本身就是3
        data = data.astype(np.float32)          # 改变数据类型
        rand_ud = rand_ud.astype(np.float32)
        
        # data = data / 255.0

        new_sample = {
        'data': torch.from_numpy(data),       # 根据输入参数构造一个相同的张量
        'rand_ud': torch.from_numpy(rand_ud),
        'label': torch.from_numpy(label).long(),
        }

        return new_sample

# custom weights initialization called on netG and netD
def weights_init_g(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_init_ucnet(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

    if type(module) == nn.GroupNorm:
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)

def weights_init_wisenet(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.normal_(module.weight.data, mean=0, std=0.01)
            #nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    #if type(module) == nn.Linear:
        #nn.init.normal_(module.weight.data, mean=0, std=0.01)
        #nn.init.constant_(module.bias.data, val=0)
        
    if type(module) == nn.BatchNorm2d:
        nn.init.constant_(module.weight.data, val=1)
        nn.init.constant_(module.bias.data, val=0)

    if type(module) == nn.GroupNorm:
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


def weights_init_covnet(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

      # nn.init.xavier_uniform_(module.weight.data)
      # nn.init.constant_(module.bias.data, val=0.2)
    # else:
    #   module.weight.requires_grad = True

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)

def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss_Cov']))

    y1 = hist['D_loss_Cov']
    y2 = hist['G_loss']
    y3 = hist['D_loss_Wise']

    plt.plot(x, y1, label='D_loss_Cov')
    plt.plot(x, y2, label='G_loss')
    plt.plot(x, y3, label='D_loss_Wise')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()



def main():
    
    opt = myParseArgs()
    try:
        os.makedirs(opt.outf)
        
    except OSError:
        pass


    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    LOG_PATH = os.path.join(opt.outf, 'model_log_'+opt.config)
    setLogger(LOG_PATH, mode = 'w')

    
    transform = transforms.Compose([ALAToTensor(),])
    dataset = ALADataset256(opt.dataroot, transforms= transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers),drop_last = True)

    netG = UNet()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    netG.apply(weights_init_g)

    ############### Add texture reward ###############
    hpf = HPF_LAP2()
    hpf = nn.DataParallel(hpf)
    hpf = hpf.cuda()
    ############### Add texture reward ###############

    if opt.netG != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netG))
        logging.info('-' * 8)
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    
    netWise = WISENet()
    netWise = nn.DataParallel(netWise)
    netWise = netWise.cuda()
    netWise.apply(weights_init_wisenet)

    if opt.netWise != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netWise))
        logging.info('-' * 8)
        netWise.load_state_dict(torch.load(opt.netWise))
    # print(netWise)

    
    netCov = CovNet()
    netCov = nn.DataParallel(netCov)
    netCov = netCov.cuda()
    netCov.apply(weights_init_covnet)
    
    if opt.netCov != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netCov))
        logging.info('-' * 8)
        netCov.load_state_dict(torch.load(opt.netCov))
    # print(netCov)


    criterion = nn.CrossEntropyLoss().cuda()
    

    # real_label = 0
    # fake_label = 1
    # setup optimizer
    optimizerD1 = optim.Adam(netWise.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(netCov.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

    scheduler_D1 = StepLR(optimizerD1, step_size=20, gamma=0.4)
    scheduler_D2 = StepLR(optimizerD2, step_size=20, gamma=0.4)
    scheduler_G = StepLR(optimizerG, step_size=20, gamma=0.4)

    iteration_Wise = 0
    iteration_Cov = 0
    
    train_hist = {}
    train_hist['D_loss_Cov'] = []
    train_hist['D_loss_Wise'] = []
    train_hist['G_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    start_time = time.time()
    for epoch in range(opt.niter):

        netG.train()
        netWise.train()
        netCov.train()

        scheduler_D1.step()
        scheduler_D2.step()
        scheduler_G.step()


        epoch_start_time = time.time()
        for i,sample in enumerate(dataloader,0):
            
            # train with real
            optimizerG.zero_grad()

            data, rand_ud, label = sample['data'], sample['rand_ud'],sample['label']
            cover, n, label = data.cuda(), rand_ud.cuda(), label.cuda()

            cover = cover[:,:1,:,:]
            n = n[:,:1,:,:]

            p_p, p_m = netG(cover)

            p_plus = p_p / 2.0 + 1e-5
            p_minus = p_m / 2.0 + 1e-5
            
            m = torch.zeros_like(cover)
            m[n < p_plus] = 1
            m[n > (torch.ones_like(cover)-p_minus)] = -1
            
            stego = cover + m
            stego.requires_grad = True
            stego.retain_grad()

            C = -(p_plus * torch.log2(p_plus) + p_minus*torch.log2(p_minus)+ (1 - p_plus - p_minus +1e-5) * torch.log2(1 - p_plus - p_minus +1e-5))

        
            d_input = torch.zeros(opt.batchSize*2,1,256,256).cuda()
            d_input[0:opt.batchSize*2:2,:] = cover
            d_input[1:opt.batchSize*2:2,:] = stego
            
            ############### Add texture reward ###############
            texture = hpf(cover)
            texture = abs(texture)
            texture = torch.sum(texture, dim = 0) 
            ############### Add texture reward ###############
            
            label = label.reshape(-1)

            ########################## update G ##########################
            optimizerD1.zero_grad()
            errD1 = criterion(netWise(d_input), label) #.detach()
            
            optimizerD2.zero_grad()
            errD2 = criterion(netCov(d_input), label) #.detach()

            ################## add DDLloss ####################
            ## number of discriminators
            numD = 2    
            errAvg = (errD1 + errD2)/numD
            errDDL = (abs(errD1-errAvg) + abs(errD2-errAvg))/(numD*(torch.exp(1-errD1)+torch.exp(1-errD2)))
            newErrD1 = errD1 + torch.exp(1-errD1)*errDDL
            newErrD2 = errD2 + torch.exp(1-errD2)*errDDL
            ################## add DDLloss ####################
            
            # 选最小的loss更新G，说明其能力更强
            err_list = [newErrD1, newErrD2]
            min_idx = err_list.index(min(err_list))

            if min_idx == 0:
                newErrD1.backward(retain_graph=True)
                m_grad = stego.grad.data #用于更新G
                reward = 1e+7 * m * m_grad
                mask_plus = torch.where(m.eq(torch.ones_like(cover)),  
                                    torch.ones_like(cover), torch.zeros_like(cover))
                mask_minus = torch.where(m.eq(-1 * torch.ones_like(cover)),
                                    torch.ones_like(cover), torch.zeros_like(cover))
                entropy = torch.log(p_plus) * mask_plus + torch.log(p_minus) * mask_minus 
                
                g_l1 = torch.mean(entropy * reward * texture)
                iteration_Wise += 1

            else:
                newErrD2.backward(retain_graph=True) 
                m_grad = stego.grad.data #用于更新G
                reward = 1e+7 * m * m_grad
                mask_plus = torch.where(m.eq(torch.ones_like(cover)),  
                                    torch.ones_like(cover), torch.zeros_like(cover))
                mask_minus = torch.where(m.eq(-1 * torch.ones_like(cover)),
                                    torch.ones_like(cover), torch.zeros_like(cover))
                entropy = torch.log(p_plus) * mask_plus + torch.log(p_minus) * mask_minus 
                g_l1 = torch.mean(entropy * reward * texture)
                iteration_Cov += 1

            g_l2 = torch.mean((C.sum(dim = (1,2,3)) - 256 * 256 * PAYLOAD) ** 2)
            errG = -g_l1 + 1e-7*g_l2
            errG.backward()
            optimizerG.step()
            ########################## update G ##########################
            

            ########################## update D ##########################
            optimizerD1.zero_grad()
            errD1 = criterion(netWise(d_input), label) #.detach()
            
            optimizerD2.zero_grad()
            errD2 = criterion(netCov(d_input), label) #.detach()

            ################## add DDLloss ####################
            numD = 2        ## number of discriminators
            errAvg = (errD1 + errD2)/numD
            errDDL = (abs(errD1-errAvg) + abs(errD2-errAvg))/(numD*(torch.exp(1-errD1)+torch.exp(1-errD2)))
            newErrD1 = errD1 + torch.exp(1-errD1)*errDDL
            newErrD2 = errD2 + torch.exp(1-errD2)*errDDL

            ################## add DDLloss ####################
            
            # 选最大的loss更新D，说明其能力更弱，同时由于增加了errDDL，因此其余判别器也进行更新
            err_list = [newErrD1, newErrD2]
            max_idx = err_list.index(max(err_list))
            
            if max_idx == 0:
                newErrD1.backward()                
            else:
                newErrD2.backward()                
            
            optimizerD1.step()
            optimizerD2.step()
            ########################## update D ##########################

            if epoch > 0:
                train_hist['G_loss'].append(errG.item())
                train_hist['D_loss_Cov'].append(errD1.item())
                train_hist['D_loss_Wise'].append(errD2.item())

            logging.info('Epoch: [%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4ff  Loss_G: %.4f  C:%.4f  g_l1:%.4f  g_l2:%.4f' % 
            (epoch, opt.niter-1, i, len(dataloader), errD1.item(), errD2.item(), errG.item(), C.sum().item()/opt.batchSize, -g_l1, 1e-7*g_l2))
           
            logging.info('Epoch: [%d/%d][%d/%d] newLoss_D1: %.4f  newLoss_D2: %.4f' % 
            (epoch, opt.niter-1, i, len(dataloader), newErrD1.item(), newErrD2.item()))

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
        # do checkpointing
        if (epoch+1)%10 == 2 and (epoch + 1) >= (opt.niter - 14):
        
            torch.save(netG.state_dict(), '%s/netG_epoch_%s_%d.pth' % (opt.outf, opt.config, epoch+1))
        
        logging.info("wisenet for %d iters, CovNet for %d iters" % (iteration_Wise, iteration_Cov) )
        loss_plot(train_hist, opt.outf, model_name = opt.outf + opt.config)

    train_hist['total_time'].append(time.time() - start_time)
    logging.info("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                            epoch, train_hist['total_time'][0]))
    logging.info("wisenet for %d iters, CovNet for %d iters" % (iteration_Wise, iteration_Cov) )
    logging.info("Training finish!... save training results")

if __name__ == '__main__':
    main()