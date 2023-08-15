import os, sys
import time
from setting import *
import torch.nn as nn
from torch.utils.data import DataLoader
from loaddata.dataset import Dataset_single
from model.generate_model import generate_backbone_model
from model.models_channel_cluster import channel_cluster_layer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import create_result_xls, index_to_label
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def cal_clustering_loss(grouping_result, target, avg=True):
    grouping_loss = torch.zeros(1).cuda()

    grouping_result = grouping_result.unsqueeze(-1)
    res_tmp = 1. - grouping_result

    grouping_label = torch.cat((grouping_result, res_tmp), dim=-1)

    for i in range(grouping_label.shape[0]):
        for j in range(target.shape[0]):
            loss = loss_func(grouping_label[i, j, :, :], target[j])
            grouping_loss += loss

    if avg:
        sample_num = grouping_label.shape[0] * grouping_label.shape[1]
        grouping_loss = grouping_loss / sample_num
    return grouping_loss

def load_cluster_result(phase):
    if sets.cluster_hierarchical is not True:
        part_num = sets.part_num
        part_index = np.load(file='{}/cluster_result_{}_{}_{}{}.npy' \
                             .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1],
                                     phase, str(part_num)))
        cluster_label = index_to_label(part_index, part_num)

    else:
        hcluster_result = torch.load('{}/hcluster_result_{}_{}_{}{}.pth'
                                     .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1],
                                             phase, sets.part_nums_hierarchical))
        cluster_label = index_to_label(hcluster_result['im2cluster'][0], sets.part_nums_hierarchical[0])

    cluster_label = np.array(cluster_label)
    cluster_label = torch.LongTensor(cluster_label)
    cluster_label.cuda()

    return cluster_label



def train(data_loader, cluster_label):
    ccl.train()
    backbone.eval()
    running_loss = 0.

    for i, sample in enumerate(data_loader):
        MRI_img, label = sample['T1'], sample['label']
        MRI_img, label = MRI_img.cuda(), label.cuda()

        _, conv_features = backbone(MRI_img)
        conv_features = conv_features.reshape(-1, sets.ccl_dim, 9, 11, 9)
        target = cluster_label.cuda()

        # compute output
        channel_grouping_res = ccl(conv_features)
        grouping_result, weighted_feature = channel_grouping_res[0], channel_grouping_res[1]

        optimizer.zero_grad()
        loss = cal_clustering_loss(grouping_result, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    ave_loss = running_loss / (i + 1)
    return ave_loss


def test(data_loader, cluster_label):
    ccl.eval()
    backbone.eval()
    running_loss = 0.

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            MRI_img, label = sample['T1'], sample['label']
            MRI_img, label = MRI_img.cuda(), label.cuda()

            _, conv_features = backbone(MRI_img)
            conv_features = conv_features.reshape(-1, sets.ccl_dim, 9, 11, 9)
            target = cluster_label.cuda()

            # compute output
            channel_grouping_res = ccl(conv_features)
            grouping_result, weighted_feature = channel_grouping_res[0], channel_grouping_res[1]

            optimizer.zero_grad()
            loss = cal_clustering_loss(grouping_result, target)

            running_loss += loss.item()
    ave_loss = running_loss / (i + 1)
    return ave_loss
def main():

    # load data
    train_dataset = Dataset_single(sets, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=True, pin_memory=True)
    test_dataset = Dataset_single(sets, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=False, pin_memory=False)

    train_cluster_label = load_cluster_result('train')


    print('training channel cluster layer')
    epochs = sets.n_epochs_ccl
    for epoch in range(epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())), end=' ')
        # print("epoch={} lr={}]".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        train_loss = train(train_loader, train_cluster_label)
        test_loss = test(test_loader, train_cluster_label)
        scheduler.step(test_loss)

        print("Epoch = {} lr = {}: Train_loss = {:.9f}, Test_loss = {:.9f}".format(epoch, lr, train_loss, test_loss))
        torch.save(ccl.state_dict(), '{}/channel_cluster_layer({})_{}.pth'
                   .format(sets.pretrain_savepath_ccl, epoch, test_loss))

    print('Finished Training')







if __name__ == '__main__':
    # settting
    sets = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sets.gpu_id)
    torch.manual_seed(sets.manual_seed)
    print("Current setting is:")
    print(sets)
    print("\n\n")

    # getting backbone model to extract convolutional features.
    # the features are flattened and need to be reshaped before next layer.
    sets.phase, sets.state = 'test', 3
    backbone = generate_backbone_model(sets)
    backbone.eval()
    print(backbone)

    # getting channel cluster layer to pretrain
    if sets.cluster_hierarchical is not True:
        part_num = sets.part_num
        sets.pretrain_savepath_ccl = '{}/{}'.format(sets.pretrain_savepath_ccl, part_num)
    else:
        part_num = sets.part_nums_hierarchical[0]
        sets.pretrain_savepath_ccl = '{}/{}'.format(sets.pretrain_savepath_ccl, sets.part_nums_hierarchical)

    # part_num = sets.part_nums_hierarchical[0] if sets.cluster_hierarchical is True else sets.part_num
    ccl = channel_cluster_layer(part_num=part_num, channel_num=sets.ccl_dim).cuda()
    if sets.model_ccl4pretrain:
        ccl.load_state_dict(torch.load(sets.model_ccl4pretrain))

    optimizer = torch.optim.Adam(ccl.parameters(), lr=sets.lr_pretrain_ccl, weight_decay=1e-7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    loss_func = nn.CrossEntropyLoss(sets.w).cuda()

    # save
    if not os.path.exists(sets.pretrain_savepath_ccl):
        os.makedirs(sets.pretrain_savepath_ccl)

    main()
