#!/usr/bin/env python
''' MENet Training code
Training code for Brain diseases classification based on T1-MRI
Written by Yilin Leng

'''

# import os
import time
# import torch
import torch.nn as nn
import torch.nn.functional as F
from setting import *
from utils.utils import setup_seed, create_result_xls, calculate
from utils.loss import channel_clustering_loss, channel_clustering_hpro_loss
from model.generate_model import generate_backbone_model
from model.models_channel_cluster import channel_cluster_layer
from model.models_gcn import dgcn_cls
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from loaddata.dataset import Dataset_single
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def train_backbone_dgcn(epoch, dataloader, running_loss_cls):
    backbone.train()
    ccl.eval()
    dgcn.train()

    for i, sample in enumerate(dataloader):
        MRI_img, label = sample['T1'], sample['label']
        MRI_img, label = MRI_img.cuda(), label.cuda()

        _, conv_features = backbone(MRI_img)
        conv_features = conv_features.reshape(-1, sets.ccl_dim, 9, 11, 9)
        grouping_result, weighted_feature = ccl(conv_features)
        cls_res = dgcn(conv_features, weighted_feature)

        backbone_optimizer.zero_grad()
        dgcn_optimizer.zero_grad()

        loss = cls_loss(cls_res, label)
        running_loss_cls += loss.item()

        loss.backward()
        dgcn_optimizer.step()

        if epoch > 5:
            backbone_optimizer.step()

        # print statistics
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_cls / (i + 1)))

    return loss


def train_ccl(epoch, dataloader, hcluster_result):
    backbone.eval()
    ccl.train()
    dgcn.eval()
    running_loss0, running_loss1 = 0., 0.

    for i, sample in enumerate(dataloader):
        MRI_img, label = sample['T1'], sample['label']
        MRI_img, label = MRI_img.cuda(), label.cuda()

        conv_features = backbone(MRI_img)
        conv_features = conv_features.reshape(-1, sets.ccl_dim, 9, 11, 9)
        grouping_result, weighted_feature = ccl(conv_features)
        cls_res = dgcn(conv_features, weighted_feature)

        ccl_optimizer.zero_grad()

        loss1 = cls_loss(cls_res, label)  # classification loss
        if sets.cluster_hierarchical is not True:
            loss2 = ccl_loss(conv_features)  # [dis_loss, div_loss]
        else:
            loss2 = ccl_loss(conv_features, hcluster_result)  # hierarchical prototype loss

        running_loss0 += loss2[0].item()
        running_loss1 += loss2[1].item()
        loss = (loss2[0] + loss2[1] + loss1)

        loss.backward()
        ccl_optimizer.step()

        if i % 50 == 49:
            print('[%d, %5d] classfication loss: %.3f' % (epoch + 1, i + 1, loss1))
            if sets.cluster_hierarchical is not True:
                print('[%d, %5d] dis/div loss: %.8f, %.8f' %
                      (epoch + 1, i + 1, running_loss0 / (i + 1), running_loss1 / (i + 1)))
            else:
                print('[%d, %5d] node/edge loss: %.8f, %.8f' %
                      (epoch + 1, i + 1, running_loss0 / (i + 1), running_loss1 / (i + 1)))
            running_loss_ccl = running_loss0 / (i + 1) + running_loss1 / (i + 1)
            scheduler_ccl.step(running_loss_ccl)

    return loss


def validate(data_loader):
    print('validating')
    backbone.eval()
    ccl.eval()
    dgcn.eval()

    val_loss = 0
    labels, predicted, predicted1 = [], [], []
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            MRI_img, label = sample['T1'], sample['label']
            MRI_img, label = MRI_img.cuda(), label.cuda()

            _, conv_features = backbone(MRI_img)
            conv_features = conv_features.reshape(-1, sets.ccl_dim, 9, 11, 9)
            grouping_result, weighted_feature = ccl(conv_features)
            cls_res = dgcn(conv_features, weighted_feature)

            pred = torch.argmax(cls_res.data, -1)
            score = F.softmax(cls_res.data, dim=1)  # 貌似不需要

            # total += target.size(0)
            # correct += (pred == target).sum().item()

            labels.extend(label.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            predicted1.extend(score[:, 1].cpu().numpy())

            del (cls_res)

    acc, auc, F1_score, sen, spe = calculate(labels, predicted, predicted1)
    print('validating acc: %3f%%'%(acc))

    return acc, auc, F1_score, sen, spe
    torch.cuda.empty_cache()




def main():

    netname = ['backbone', 'ccl', 'gnn']
    path_dir = dict()
    for p in range(3):
        path_dir[p] = r'{}/{}'.format(sets.sum_savpath, netname[p])
        if not os.path.exists(path_dir[p]):
            os.makedirs(path_dir[p])

    file, xls_name, sheet = create_result_xls(sets.sum_savpath, sets.classes, sets)
    writer_path = sets.log_savpath
    writer = SummaryWriter(writer_path)
    print('RESULTS SHOW IN : tensorboard --logdir={}'.format(writer_path))


    # load data
    train_dataset = Dataset_single(sets, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=True, pin_memory=True)
    test_dataset = Dataset_single(sets, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=False, pin_memory=False)
    hcluster_result = torch.load('{}/hcluster_result_{}_{}_{}{}.pth'
                                 .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1],
                                         'train', sets.part_nums_hierarchical))
    # cluster_label = index_to_label(hcluster_result['im2cluster'][0], sets.part_nums_hierarchical[0])

    for epoch in range(sets.n_epochs_gcn):
        running_loss_cls = 0.

        print('training')
        if (epoch+1) % sets.n_epoch_interval != 0:
            train_loss = train_backbone_dgcn(epoch, train_loader, running_loss_cls)
        else:
            train_loss = train_ccl(epoch, train_loader, hcluster_result)

        test_acc, test_auc, test_f1, test_sen, test_spe = validate(test_loader)


    writer.add_scalar('train_loss', train_loss.item(), epoch)
    writer.add_scalars('test_acc_auc', {'acc': test_acc.item(), 'auc': test_acc.item() * 100}, epoch)
    writer.add_scalars('test_sen_spe', {'sen': test_sen.item(), 'spe': test_spe.item()}, epoch)

    save_result = 'epoch{}_acc{:.2f}_auc{:.4f}_sen{:.4f}_spe{:.4f}'.format(epoch,test_acc,test_acc,test_sen,test_spe)
    if (epoch + 1) % sets.n_epoch_interval != 0:
        print('saving cls checkpoints....')
        torch.save(dgcn.state_dict(), '{}/dgcn_{}.pth'.format(path_dir[2], save_result))
        if epoch > 5:
            torch.save(backbone.state_dict(), '{}/{}_{}.pth'.format(path_dir[0], sets.backbone, save_result))
    else:
        print('saving cgl checkpoints....')
        torch.save(ccl.state_dict(), '{}/ccl_{}.pth'.format(path_dir[1], save_result))

    # save results to excel
    datalist = [epoch, dgcn_optimizer.param_groups[0]['lr'],train_loss.item(),
                test_acc, test_auc, test_f1, test_sen, test_spe]
    for i in range(0, 8):
        sheet.write(epoch + 1, i, datalist[i])
    file.save(xls_name)

    torch.cuda.empty_cache()













if __name__ == '__main__':
    # settting
    setup_seed(9)
    sets = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sets.gpu_id)
    torch.manual_seed(sets.manual_seed)
    print("Current setting is:")
    print(sets)
    print("\n\n")

    sets.phase, sets.state = 'train', 4
    if sets.cluster_hierarchical is not True:
        part_num = sets.part_num
        sets.pretrain_savepath_ccl = '{}/{}'.format(sets.pretrain_savepath_ccl, part_num)
        ccl_loss = channel_clustering_loss()
    else:
        part_num = sets.part_nums_hierarchical[0]
        sets.pretrain_savepath_ccl = '{}/{}'.format(sets.pretrain_savepath_ccl, sets.part_nums_hierarchical)
        ccl_loss = channel_clustering_hpro_loss(sets.part_nums_hierarchical)

    part_num = sets.part_num #houmianshandiao
    present_time = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
    sets.sum_savpath = '{}/cluster{}/{}'.format(sets.sum_savpath, part_num, present_time)
    sets.log_savpath = '{}/{}/{}'.format(sets.log_savpath, part_num, present_time)

    backbone = generate_backbone_model(sets).cuda()
    ccl = channel_cluster_layer(part_num=part_num, channel_num=sets.ccl_dim).cuda()
    ccl.load_state_dict(torch.load(sets.model_ccl))
    dgcn = dgcn_cls(part_num=part_num).cuda()

    # loss function
    cls_loss = nn.CrossEntropyLoss(weight=sets.w).cuda()
    ccl_loss = ccl_loss.cuda()

    # optimizer and scheduler
    backbone_optimizer = torch.optim.Adam(backbone.parameters(), lr=sets.lr_backbone, weight_decay=sets.weight_decay)
    ccl_optimizer = torch.optim.Adam(ccl.parameters(), lr=sets.lr_ccl, weight_decay=sets.weight_decay)
    dgcn_optimizer = torch.optim.Adam(dgcn.parameters(), lr=sets.lr_gcn, weight_decay=sets.weight_decay)

    scheduler_backbone = ReduceLROnPlateau(backbone_optimizer, mode='min', factor=0.1, patience=1)
    scheduler_ccl = ReduceLROnPlateau(ccl_optimizer, mode='min', factor=0.1, patience=3)
    scheduler_dgcn = ReduceLROnPlateau(dgcn_optimizer, mode='min', factor=0.1, patience=1)


    main()







