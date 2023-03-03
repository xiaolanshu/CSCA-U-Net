import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
from utils.dataloader import test_dataset
from collections import OrderedDict
from skimage import img_as_ubyte
import time

#  import models ....
from model.CSCAUNet import CSCAUNet


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/CSCAUNet-bs16-lr1e4-DRIVE/CSCAUNet-bs16-lr1e4-DRIVE.pth')
parser.add_argument('--test_path',type=str)
parser.add_argument('--train_save', type=str,default='')


for _data_name in ['test','CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    opt = parser.parse_args()
    targetdir=os.listdir(opt.test_path)
    if(_data_name not in targetdir):
        continue

    data_path = os.path.join(opt.test_path,_data_name)
    print("#"*20)
    print("Now test dir is: ",data_path)
    print("#"*20)
    time.sleep(10)

    save_path = './Result/'+opt.train_save+'/{}/{}/'.format(opt.test_path.split('/')[-2],_data_name)
    
    model =CSCAUNet(1,2)
    pth_path=os.path.join('./snapshots',opt.train_save,opt.train_save+'.pth')
    weights = torch.load(pth_path)
    new_state_dict = OrderedDict()

    for k, v in weights.items():
        if 'total_ops' not in k and 'total_params' not in k:
            name = k
            new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()


    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        res5,res4,res3,res2,res1,res0 = model(image)
        res = res5
        print(res.shape)
        res = F.interpolate(res, size=(gt.shape), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        imageio.imsave(save_path+name, img_as_ubyte(res))
