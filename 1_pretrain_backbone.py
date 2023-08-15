#!/usr/bin/env python
''' MENet Training code
Training code for Brain diseases classification based on T1-MRI
Written by Yilin Leng

'''

import os
import xlwt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loaddata.dataset import Dataset_single
from setting import *
from model.generate_model import generate_backbone_model
from utils.utils import create_result_xls, calculate

import warnings
warnings.filterwarnings("ignore")


def train(train_loader):

    correct, predicted, predicted1 = [], [], []
    train_loss = 0
    net.cuda().train()

    for i, sample in enumerate(train_loader):
        MRI_img, label = sample['T1'], sample['label']
        MRI_img, label = MRI_img.cuda(), label.cuda()

        optimizer.zero_grad()
        embeddings, outputs = net(MRI_img)

        _, pred = torch.max(outputs.data, 1)
        score = F.softmax(outputs.data, dim=1)
        outputs = outputs.float().cuda()
        loss_cross = lossfunction_cross(outputs, label)

        loss = loss_cross

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        correct.extend(label.cpu().numpy())
        predicted.extend(pred.cpu().numpy())
        predicted1.extend(score[:, 1].cpu().numpy())

    scheduler.step()
    del MRI_img, label, outputs, sample
    torch.cuda.empty_cache()

    acc, auc, F1_score, sen, spe = calculate(correct, predicted, predicted1)
    return train_loss/len(train_loader), acc, auc, F1_score, sen, spe
    exit()


def test(test_loader):

    net.cuda().eval()
    test_loss = 0
    correct, predicted, predicted1 = [], [], []

    for batch_idx, sample in enumerate(test_loader):
        MRI_img, label = sample['T1'], sample['label']
        MRI_img, label = MRI_img.cuda(), label.cuda()

        with torch.no_grad():
            embeddings, outputs = net(MRI_img)

            _, pred = torch.max(outputs.data, 1)
            score = F.softmax(outputs.data, dim=1)
            outputs = outputs.float().cuda()

            test_cross = lossfunction_cross(outputs, label)
            testloss = test_cross
            test_loss += testloss.item()

            correct.extend(label.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            predicted1.extend(score[:, 1].cpu().numpy())
    torch.cuda.empty_cache()

    acc, auc, F1_score, sen, spe = calculate(correct, predicted, predicted1)
    return test_loss / len(test_loader), acc, auc, F1_score, sen, spe
    exit()





def main():

    # load data
    train_dataset = Dataset_single(sets, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=True,pin_memory=True)
    test_dataset = Dataset_single(sets, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=False, pin_memory=False)


    # settings
    print("Current setting is:")
    print(sets)
    print("\n\n")

    # saveauc, saveacc = 0.5, 55
    for epoch in range(sets.n_epochs_backbone):
        print(time.strftime('[ %m-%d %H:%M:%S', time.localtime(time.time())), end=' ')
        print("epoch={} lr={}]".format(epoch, scheduler.get_last_lr()[0]))

        # training
        train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1 = train(train_loader)
        print("Train: loss = {:.4f}, acc = {:.2f}%, auc = {:.4f}, F1_score = {:.4f}, sen = {:.4f}, spe = {:.4f}"
              .format(train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1))

        # testing
        if train_acc > 60:
            test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2 = test(test_loader)
            print("Test : loss = {:.4f}, acc = {:.2f}%, auc = {:.4f}, F1_score = {:.4f}, sen = {:.4f}, spe = {:.4f}"
                  .format(test_loss, test_acc, test_auc, test_F1_socre,sen2, spe2))

            # save model
            if test_auc > sets.saveauc or test_acc > sets.saveacc:
                sets.saveauc, sets.saveacc, savesen, savespe = int(test_auc * 10000) / 10000, int(test_acc * 100) / 100, int(
                    sen2 * 100) / 100, int(spe2 * 100) / 100
                model_save_path = '{}/epoch{}_acc{}_auc{}_sen{}_spe{}_{}_{}.pth'\
                    .format(sets.pretrain_savepath_backbone, epoch, sets.saveacc, sets.saveauc, savesen, savespe, sets.classes[0], sets.classes[1])
                torch.save(net.state_dict(), model_save_path)
        else:
            test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2 = 0, 0, 0, 0, 0, 0
            print("train_acc < 0.6")


        # save results to excel
        datalist = [epoch, scheduler.get_last_lr()[0],
                    train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1,
                    test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2]
        for i in range(0, 14):
            sheet.write(epoch + 2, i, datalist[i])
        file.save(file_name)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sets.gpu_id)
    torch.manual_seed(sets.manual_seed)

    # getting model
    sets.phase, sets.state = 'train', 1 # 0: backbone pretrain; 1: cluster; 2: ccl pretrain; 3: backbone+gcn train
    net, parameters = generate_backbone_model(sets)
    print(net)

    # optimizer
    if sets.model_backbone4pretrain:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.lr_pretrain_backbone},
            {'params': parameters['new_parameters'], 'lr': sets.lr_pretrain_backbone * 100}
        ]
    else:
        params = [{'params': parameters, 'lr': sets.lr_pretrain_backbone * 100}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 350, 500, 700], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    weight = torch.from_numpy(np.array(sets.w)).float().cuda()
    lossfunction_cross = nn.CrossEntropyLoss(weight=weight)


   # save
   # pretrain_savepath_backbone = sets.pretrain_savepath_backbone
    if not os.path.exists(sets.pretrain_savepath_backbone):
        os.makedirs(sets.pretrain_savepath_backbone)
    file, file_name, sheet = create_result_xls(sets.pretrain_savepath_backbone, sets.classes, sets)


    main()
