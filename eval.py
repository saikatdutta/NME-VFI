import argparse
import os
import os.path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from torch.utils.data import Dataset
from tqdm import tqdm

device = torch.device('cuda:0')

from model import InterpNet

#####################################################################################################

def to_image(im_tensor):
    im_tensor = torch.clamp(im_tensor,0,1)
    im_tensor = im_tensor.detach().cpu().squeeze()
    im = np.transpose(im_tensor.numpy(),(1,2,0))
    im = im[:,:,::-1]
    im = np.uint8(im*255)

    return im 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class FrameDataset(Dataset):

    def __init__(self, csv_file, data_root = './', transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data.iloc[idx,5] 

        frame0 = Image.open(os.path.join(self.data_root,self.data.iloc[idx, 0]))
        frame1 = Image.open(os.path.join(self.data_root,self.data.iloc[idx, 1]))
        frame2 = Image.open(os.path.join(self.data_root,self.data.iloc[idx, 2])) 
        frame3 = Image.open(os.path.join(self.data_root,self.data.iloc[idx, 3]))
        framet = Image.open(os.path.join(self.data_root,self.data.iloc[idx, 4])) 


        if self.transform:
            frame0 = self.transform(frame0)
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)
            framet = self.transform(framet) 

        return (frame0, frame1, frame2, frame3, framet, t)


######################################################################################################  

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--data_root', type=str, default= './')
args = parser.parse_args()


synnet = InterpNet().to(device)
synnet.load_state_dict(torch.load('checkpoints/main/ckpt.pth'))
print ('Total model parameters: ', count_parameters(synnet)/1e6, ' M')   

transform_pwc = transforms.Compose([ transforms.ToTensor()])

if args.dataset == 'vimeo':
    testset = FrameDataset(csv_file = 'datasets/vimeo_quintuplets.csv', data_root = args.data_root, transform = transform_pwc)
elif args.dataset == 'davis':
    testset = FrameDataset(csv_file = 'datasets/davis_quintuplets.csv', data_root = args.data_root, transform = transform_pwc)
elif args.dataset == 'gopro':    
    testset = FrameDataset(csv_file = 'datasets/gopro_quintuplets.csv', data_root = args.data_root, transform = transform_pwc)
# elif args.dataset == 'hd':
# testset = FrameDataset(csv_file = 'datasets/HDD_quintuplets.csv', data_root = args.data_root, transform = transform_pwc)
else :
    print ("Invalid dataset. Exiting ...")
    sys.exit(0)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)


outdir = os.path.join('outputs/', args.dataset)
os.makedirs(outdir, exist_ok=True)

total_psnr = 0 
total_ssim = 0
count = 0

with torch.no_grad():

    for i,data in enumerate(tqdm(testloader),0):
        
        I0, I1 , I2, I3 , It, timestamp = data     # It lies within I1 and I2
        It_im = to_image(It)

        I0 = I0.to(device)
        I1 = I1.to(device)
        I2 = I2.to(device)
        I3 = I3.to(device)
        
        It_pred_list = synnet(I0, I1, I2, I3)

        It_pred = It_pred_list[0]

        It_pred = to_image(It_pred)

        cv2.imwrite(os.path.join(outdir, str(i)+ '.png'), It_pred)

        psnr = compare_psnr(It_im, It_pred)
        ssim = compare_ssim(It_im, It_pred, multichannel=True)

        
        total_psnr += psnr 
        total_ssim += ssim
        count += 1
         

print ('Avg PSNR: ', total_psnr/count)
print ('Avg SSIM: ', total_ssim/count)

