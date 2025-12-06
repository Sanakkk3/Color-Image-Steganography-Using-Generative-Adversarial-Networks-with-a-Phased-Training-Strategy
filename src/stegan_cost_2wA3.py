import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import os


from PIL import Image
import time
import cv2
import scipy.io as sio
from pathlib import Path

from ResUNet1DB import UNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

WETCOST = 10e+10

def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='./asym/netG_epoch_G7_cov_wise_ucC1_DDLloss_withW_RBG_pub_72.pth', help="path to netG1 (to continue training)")
    
    parser.add_argument('--config', default='ALASKA_cost_2w_0.4_G7_cov_wise_ucC1_DDLloss_withW_RBG_pubA', help="train result")
    parser.add_argument('--stego_config', default='ALASKA_stegos_2w_0.4_G7_cov_wise_ucC1_DDLloss_withW_RBG_pubA', help="train result")

    parser.add_argument('--datacover', help='path to dataset',default='/datacover_root', help="path to testing cover images ") # testing set (20,000 images)
    parser.add_argument('--datastegoC1', help='path to dataset',default='/data/ALASKA_cost_2w_0.4_G7_cov_wise_DDLloss_withW_R_pubA/')
    parser.add_argument('--datastegoC2', help='path to dataset',default='/data/ALASKA_cost_2w_0.4_G7_cov_wise_DDLloss_withW_RB_pubA/')
    parser.add_argument('--indexpath', help='path to index',default='./index_list/alaska_train_2w.npy')

    parser.add_argument('--root', default='/data/', help="path to save result")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=25, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    
    parser.add_argument('--TS_PARA', type=int, default=1000000, help='parameter for double tanh')
    parser.add_argument('--payload', type=float, default=0.4, help='embeding rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    args = parser.parse_args()
    return args

def ternary_entropyf(pP1, pM1):
    p0 = 1-pP1-pM1
    P = torch.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*torch.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return torch.sum(H)

def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0

    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (torch.exp(-l3 * rho_p1)) / (1 + torch.exp(-l3 * rho_p1) + torch.exp(-l3 * rho_m1))
        pM1 = (torch.exp(-l3 * rho_m1)) / (1 + torch.exp(-l3 * rho_p1) + torch.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0

    # iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<30:  #300
        lamb = l1+(l3-l1)/2
        pP1 = (torch.exp(-lamb*rho_p1))/(1+torch.exp(-lamb*rho_p1)+torch.exp(-lamb*rho_m1))
        pM1 = (torch.exp(-lamb*rho_m1))/(1+torch.exp(-lamb*rho_p1)+torch.exp(-lamb*rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1
    return lamb

def embedding_simulator(batch_x, batch_rho_p1, batch_rho_m1, m, seed):
    batch_y = torch.zeros(batch_x.shape).cuda()
    for i in range(batch_x.shape[0]):
        x = batch_x[i,0,:,:]
        rho_p1 = batch_rho_p1[i,0,:,:]
        rho_m1 = batch_rho_m1[i, 0, :, :]

        n = x.shape[0]*x.shape[1]
        lamb = calc_lambda(rho_p1, rho_m1, m, n)
        pChangeP1 = (torch.exp(-lamb * rho_p1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))
        pChangeM1 = (torch.exp(-lamb * rho_m1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))

        y = x.clone()
        #print("seed: "+str(seed[i]))
        torch.manual_seed(seed[i])
        randChange = torch.rand([y.shape[0], y.shape[1]]).cuda()

        y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
        y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1
        batch_y[i,0,:,:] = y
    
    # batch_y = batch_y.astype(np.float32)
    # batch_y = torch.from_numpy(batch_y)
    return batch_y



# 彩色图像处理
class ALADataset256(data.Dataset):
    def __init__(self, root, c1root, c2root, index_path, transforms = None):
        self.index_list = np.load(index_path)

        self.cover_path = root + '{}.ppm'
        self.c1_path = c1root + '{}.pgm'
        self.c2_path = c2root + '{}.pgm'

        self.transforms = transforms

    def __getitem__(self, index):
        file_index = self.index_list[index]
        cover_path = self.cover_path.format(file_index)
        label = np.array([0, 1], dtype='int32')     # 作用？定义一个名为‘label’的Numpy数据，内容是[0,1]，数据内容是int32，表示‘label’变量中包含两个类别的标签或目标值，即类别0和类别1.

        data = cv2.imread(cover_path, -1)         # 8位深度，原通道  
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)    # 读进来是BGR所以要转换一下格式为RGB
        data = np.transpose(data, (2,0,1))      # 需要将原始数据转化为模型能够接受的Tensor格式 (C, H, W)
        # rand_ud = np.random.rand(3, 256, 256)   # 彩色图像是3通道，加噪声对应通道数应该也是3吧
        
        #### 因为只处理单通道，生成单通道的cost
        data = data[1:2,:,:] 

        c1_path = self.c1_path.format(file_index)
        stegoC1 = cv2.imread(c1_path, -1)

        c2_path = self.c2_path.format(file_index)
        stegoC2 = cv2.imread(c2_path, -1)

        rand_ud = np.random.rand(256, 256)
        sample = {'data': data, 'rand_ud': rand_ud, 'label': label, 'index': file_index, 'stegoC1': stegoC1, 'stegoC2':stegoC2}
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
    def __len__(self):
        return len(self.index_list)

class ALAToTensor():
    def __call__(self, sample):
        data, rand_ud, label, index, stegoC1, stegoC2= sample['data'], sample['rand_ud'], sample['label'], sample['index'], sample['stegoC1'], sample['stegoC2']

        # 读彩色图像不需要，因为维度本身就是3
        # data = np.expand_dims(data, axis=0)     # 在相应的axis上扩展维度 （1,...）
        data = data.astype(np.float32)          # 改变数据类型

        stegoC1 = np.expand_dims(stegoC1, axis=0)
        stegoC1 = stegoC1.astype(np.float32)

        stegoC2 = np.expand_dims(stegoC2, axis=0)
        stegoC2 = stegoC2.astype(np.float32)
        
        rand_ud = np.expand_dims(rand_ud,axis = 0)      # 但是彩色图像是三通道，所以增加一个维度，需要改成3，在自定义数据上改了
        rand_ud = rand_ud.astype(np.float32)

        new_sample = {
        'data': torch.from_numpy(data),       # 根据输入参数构造一个相同的张量
        'rand_ud': torch.from_numpy(rand_ud),
        'label': torch.from_numpy(label).long(),
        'index': index,
        'stegoC1': stegoC1,
        'stegoC2':stegoC2,
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



def main():
    
    args = myParseArgs()
    try:
        cost_path = os.path.join(args.root, args.config)
        os.makedirs(cost_path)
    except OSError:
        pass

    try:
        stego_path = os.path.join(args.root, args.stego_config)
        os.makedirs(stego_path)
    except OSError:
        pass

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True
        
    
    transform = transforms.Compose([ALAToTensor(),])
    dataset = ALADataset256(args.datacover, args.datastegoC1, args.datastegoC2, args.indexpath, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers),drop_last = True)

    netG = UNet(in_ch=3)
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    #netG.apply(weights_init_g)
    
    if args.netG != '':
        print('Load netG state_dict in {}'.format(args.netG))
        netG.load_state_dict(torch.load(args.netG))
    

    netG.eval()
    with torch.no_grad():
        for i,sample in enumerate(dataloader,0):

            cover, rand_ud, label, index, stegoC1, stegoC2= sample['data'], sample['rand_ud'], sample['label'], sample['index'], sample['stegoC1'], sample['stegoC2']
            cover, n, label, stegoC1, stegoC2= cover.cuda(), rand_ud.cuda(), label.cuda(), stegoC1.cuda(), stegoC2.cuda()

            # learn the probas
            input = torch.cat([stegoC1, stegoC2, cover], dim=1)
            p_p, p_m = netG(input)
            
            p_plus = p_p/2.0 + 1e-5
            p_minus = p_m/2.0 + 1e-5
            
            rho_plus = torch.log(1/p_plus-2)
            rho_minus = torch.log(1/p_minus-2)
            
            ######################## STC embedding #############
            message_length = args.imageSize*args.imageSize*args.payload

            rho_plus[rho_plus > WETCOST] = WETCOST
            rho_plus[torch.isinf(rho_plus)] = WETCOST
            rho_minus[rho_minus > WETCOST] = WETCOST
            rho_minus[torch.isinf(rho_minus)] = 10e+10

            rho_plus[cover==255] = WETCOST
            rho_minus[cover==0] = WETCOST

            stego = embedding_simulator(cover, rho_plus, rho_minus, message_length , index)
            m = stego - cover

            print('batchSize num %d ’s changerate : %f '%(i, torch.mean(abs(m))))
            ######################## STC embedding #############
            
            stego[stego == -1] = 0
            stego[stego == 256] = 255

            # save cost images
            for k in range(0,stego.shape[0]):
                cost_p1 = rho_plus[k,0].detach().cpu().numpy()
                cost_p1[np.isinf(cost_p1)] = 10e+10
                cost_m1 = rho_minus[k,0].detach().cpu().numpy()
                cost_m1[np.isinf(cost_m1)] = 10e+10
                sio.savemat( ('%s%s/%d_cost.mat'%(args.root, args.config, index[k])), mdict={'cost_p1': cost_p1, 'cost_m1': cost_m1})

                img = stego[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                # cv2.imwrite('%s%s/%d.pgm' %(args.root, args.config, index[k]), img)

                modify = (torch.round(m)+2)%2*255
                modify = modify[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_modify.png' %(args.root, args.config, index[k]), modify)

                pro_m1 = p_minus*255
                probas_m1 = pro_m1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_m1)

                pro_p1 = p_plus*255
                probas_p1 = pro_p1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_p1)
                
                im = Image.fromarray(np.uint8(img[:,:,0]))
                im.save( ('%s%s/%d.pgm' %(args.root, args.config, index[k])))

                mod = Image.fromarray(np.uint8(modify[:,:,0]))
                p_m1 = Image.fromarray(np.uint8(probas_m1[:,:,0]))
                p_p1 = Image.fromarray(np.uint8(probas_p1[:,:,0]))

                imslst = [mod, p_p1, p_m1, im]
                result = Image.new('RGBA', (args.imageSize * len(imslst), args.imageSize))
                for i, im in enumerate(imslst):
                    result.paste(im, box=(i * args.imageSize , 0))
                result.save(('%s%s/%d.png' %(args.root, args.config, index[k])))

            # save final stego images
            stego_r = stegoC1
            stego_g = stego
            stego_b = stegoC2
            stego = torch.cat([stego_r, stego_g, stego_b], dim=1)
            for k in range(0, stego.shape[0]):
                img = stego[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('%s%s/%d.ppm'%(args.root, args.stego_config, index[k]), img)

    print('Stegan down for {}'.format(args.datacover))
    print('Output path {}'.format(args.root + args.config))
            

if __name__ == '__main__':
    main()