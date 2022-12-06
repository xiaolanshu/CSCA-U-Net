# -*- coding: utf-8 -*-
import time
from tabnanny import verbose
import wandb
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchstat import stat

### import models ...
from model.CSCAUNet import CSCAUNet


def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()


def adjust_learnrate(optims, lr):
    for param_groups in optims.param_groups:
        param_groups['lr'] = lr


def test(model, path):

    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    # print("test path : ",data_path)
    #####                                             #####

    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b = 0.0
    print('[test_size]', test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2, res1, res0 = model(image)
        res = res5
        res = F.interpolate(res, size=gt.shape,
                            mode='bilinear', align_corners=False)
        #  F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))

        intersection = (input_flat*target_flat)

        loss = (2 * intersection.sum() + smooth) / \
            (input.sum() + target.sum() + smooth)

        a = '{:.4f}'.format(loss)
        a = float(a)
        b = b + a

    return b/test_loader.size


def train(name, train_loader, model, optimizer, epoch, test_path):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record0, loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(
    ), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0 = model(
                images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            loss0 = structure_loss(lateral_map_0, gts)

            loss = loss5+loss4+loss3+loss2+loss1+loss0
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:

                loss_record5.update(loss5.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record0.update(loss0.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 1 == 0:
        meandice = test(model, test_path)
        wandb.log({
            "mean dice": meandice,
            "Train loss": loss.item(),
            "loss5": loss_record5.show(),

        })
        fp = open('log/log-'+name+'.txt', 'a')
        fp.write(str(meandice)+'\n')
        fp.close()

        fp = open('log/best-'+name+'.txt', 'r')
        best = fp.read()
        fp.close()

        if meandice > float(best):
            wandb.run.summary["best meandice"] = meandice
            wandb.run.summary["best epoch"] = epoch
            fp = open('log/best-'+name+'.txt', 'w')
            fp.write(str(meandice))
            fp.close()
            fp = open('log/best-'+name+'.txt', 'r')
            best = fp.read()
            fp.close()
            torch.save(model.state_dict(), save_path + name+'.pth')
            print('[Saving Snapshot:]', save_path +
                  name+'.pth', meandice, '[best:]', best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=10, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=6, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./TrainDataset/', help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./TestDataset/CVC-300/', help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='CSCAUNet-kav-best')
    parser.add_argument('--model_name', type=str,
                        help='please input your model')
    opt = parser.parse_args()

    # ---- build models ----
    # set ur gpu device, if you run by .sh, it is can comment.
    torch.cuda.set_device(0)
    A = [CSCAUNet(1, 2)]
    num = -1
    for i in range(len(A)):
        tmp = A[i].__class__.__name__
        if tmp == opt.model_name:
            num = i
            break
    print(num)
    if num == -1:
        raise RuntimeError('model Error!')
    model = A[num].cuda()
    name = model.__class__.__name__

    ################# wandb #################
    # change entity to your entity. register: https://wandb.ai
    # run=wandb.init(project=name,entity='',name=opt.train_save)
    # or by Anoymous
    run = wandb.init(project=name, name=opt.train_save, anonymous='allow')
    #########################################

    flname = opt.train_save
    print("train: ", name)
    fp = open('log/best-'+flname+'.txt', 'w')
    fp.write('0\n')
    fp.close()

    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(
            params, opt.lr, weight_decay=1e-4, momentum=0.9)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    print("#"*20, "Train params", "#"*20)
    print("model: ", model.__class__.__name__)
    print("batch size: ", opt.batchsize)
    print("train size: ", opt.trainsize)
    print("init lr: ", opt.lr)
    print("train path : ", opt.train_path)
    print("test path : ", opt.test_path)
    print("#"*20, "Training", "#"*20)
    time.sleep(10)

    for epoch in range(1, opt.epoch):
        lr = opt.lr
        if epoch > 60:
            lr = opt.lr/10
        adjust_learnrate(optimizer, lr)
        print("epoch:", epoch)
        print("lr: ", optimizer.param_groups[0]['lr'])
        train(flname, train_loader, model, optimizer, epoch, opt.test_path)
