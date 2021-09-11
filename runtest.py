import os,argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from models.net import dehaze
import os
from skimage.measure import compare_ssim,compare_psnr
from torchvision.transforms import ToTensor,ToPILImage
import time

parser=argparse.ArgumentParser()
parser.add_argument('--test_imgs',type=str,default='testpath/',help='Test imgs folder')
parser.add_argument('--ck',type=str,default='path/***.pk',help='Test imgs folder')
parser.add_argument('--dir',type=str,default='path/',help='Test imgs folder')

opt=parser.parse_args()
if not os.path.exists(opt.dir):
    os.mkdir(opt.dir)
gps=3
blocks=19
img_dir=opt.test_imgs+'/'

model_dir=opt.ck
device='cuda' #if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir)
net=dehaze()
net=net.to(device)
net=nn.DataParallel(net)
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in ckp['model'].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
#     load params
net.load_state_dict(ckp['model'])
# net = net.cpu()
net.eval()
ssims=[]
psnr2=[]
print(img_dir)
for im_name in os.listdir(img_dir+'data/'):
    print(f'\r {im_name}',end='',flush=True)
    # print(im_name)
    im_name_i = img_dir+'data/'+im_name
    haze = Image.open(im_name_i)
    clear_na = im_name.split('/')[-1].split('_')[0]+'.png'
    clear = Image.open(img_dir+'label/' + clear_na)
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    with torch.no_grad():
        haze1 = haze1.cuda()
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    im = ToPILImage()(pred.clamp(0,1).cpu().data[0])

    im.save(opt.dir+im_name)
    ssims.append(compare_ssim(np.array(im), np.array(clear), multichannel=True))
    psnr2.append(compare_psnr(np.array(im), np.array(clear)))

print(np.mean(ssims), np.mean(psnr2))