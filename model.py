import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F 

from gridnet import GridNet
from flow_reversal import FlowReversal
from collections import OrderedDict

from gridnet3d import Grid3DNet

import sys
import cv2
import torchvision

sys.path.insert(1,'/media/data/saikat/irr/')
from models.pwcnet_occ_bi import PWCNet


def convert_ckpt(state_dict):

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict   


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)

    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut

class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x

class InterpNet(nn.Module):
    def __init__(self):

        super(InterpNet, self).__init__()

        self.fnet = Grid3DNet(6 , 32, channel_list=[16,32,64])
        self.flownet = PWCNet()
        
        self.flo_layer = nn.Conv3d(32, 4 , (3,3,3) , (1,1,1), (0,1,1) )
        self.refinenet = GridNet(16,8)
        self.masknet = SmallMaskNet(38,1)
        self.fwarp = FlowReversal()


    def forward(self,  I0_in, I1_in, I2_in, I3_in, t=0.5):

        B,C,H,W = I0_in.shape

        if (H%64==0):
            h1 = H
        else:
            h1 = ((H//64)+1)*64
        

        if (W%64==0):
            w1 = W
        else:
            w1 = ((W//64)+1)*64

        I0 = F.interpolate(I0_in, (h1,w1), mode='bilinear')
        I1 = F.interpolate(I1_in, (h1,w1), mode='bilinear')
        I2 = F.interpolate(I2_in, (h1,w1), mode='bilinear')
        I3 = F.interpolate(I3_in, (h1,w1), mode='bilinear')
        
        out01 = self.flownet({'input1':I0,'input2':I1})
        out12 = self.flownet({'input1':I1,'input2':I2})
        out23 = self.flownet({'input1':I2,'input2':I3})

        F_0_1 = out01['flow_f']
        occ_0_1 = torch.sigmoid(out01['occ_f'])

        F_1_0 = out01['flow_b']
        occ_1_0 = torch.sigmoid(out01['occ_b'])

        F_1_2 = out12['flow_f']
        occ_1_2 = torch.sigmoid(out12['occ_f'])

        F_2_1 = out12['flow_b']
        occ_2_1 = torch.sigmoid(out12['occ_b'])

        F_2_3 = out23['flow_f']
        occ_2_3 = torch.sigmoid(out23['occ_f'])

        F_3_2 = out23['flow_b']
        occ_3_2 = torch.sigmoid(out23['occ_b'])

        F_0_x = torch.zeros_like(F_0_1)
        occ_0_x = torch.zeros_like(occ_0_1)

        F_3_y = torch.zeros_like(F_0_1)
        occ_3_y = torch.zeros_like(occ_0_1)

        feat0 = torch.cat([F_0_x, occ_0_x, F_0_1, occ_0_1],dim=1)
        feat1 = torch.cat([F_1_0, occ_1_0, F_1_2, occ_1_2],dim=1)
        feat2 = torch.cat([F_2_1, occ_2_1, F_2_3, occ_2_3],dim=1)
        feat3 = torch.cat([F_3_2, occ_3_2, F_3_y, occ_3_y],dim=1)

        input_feats = torch.stack([feat0, feat1, feat2, feat3],dim=2)

        feats = self.fnet(input_feats) 
        flo_rep = self.flo_layer(feats) 

        F_1_t = flo_rep[:,0:2,0,:,:]*t + flo_rep[:,2:4,0,:,:]*t**2
        F_2_t = flo_rep[:,0:2,1,:,:]*(1-t) + flo_rep[:,2:4,1,:,:]*(1-t)**2 


        # Flow Reversal
        Ft1, norm1 = self.fwarp(F_1_t, F_1_t)
        Ft1 = -Ft1
        Ft2, norm2 = self.fwarp(F_2_t, F_2_t)
        Ft2 = -Ft2

        Ft1[norm1 > 0] = Ft1[norm1 > 0]/norm1[norm1>0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0]/norm2[norm2>0].clone()

        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, Ft1, Ft2], dim=1))

        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10*torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10*torch.tanh(output[:, 6:8])) + output[:, 2:4]

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

        Ft1r = F.interpolate(Ft1r, (H,W), mode='bilinear')
        Ft2r = F.interpolate(Ft2r, (H,W), mode='bilinear')

        Ft1r[:,0,:,:] = Ft1r[:,0,:,:]*(W/w1)
        Ft1r[:,1,:,:] = Ft1r[:,1,:,:]*(H/h1)

        Ft2r[:,0,:,:] = Ft2r[:,0,:,:]*(W/w1)
        Ft2r[:,1,:,:] = Ft2r[:,1,:,:]*(H/h1)

        M = F.interpolate(M, (H,W), mode='bilinear')

        I1tf = backwarp(I1_in, Ft1r)
        I2tf = backwarp(I2_in, Ft2r)

        Ft_p = ((1-t) * M * I1tf + t * (1 - M) * I2tf) / ((1-t) * M + t * (1-M))


        return Ft_p   

