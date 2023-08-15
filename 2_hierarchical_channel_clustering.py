#!/usr/bin/env python
''' MENet Training code
Training code for Brain diseases classification based on T1-MRI
Written by Yilin Leng
'''

import sys, os
import glob
import numpy as np
import pandas as pd
import random
from torch.nn import functional as F
from sklearn.cluster import KMeans
from model.generate_model import generate_backbone_model
from utils.gradcam import generate_camapping
from loaddata.dataset import Get_MRI_data
from setting import *
from utils.kmeans import run_hkmeans, run_kmeans

import warnings
warnings.filterwarnings("ignore")



features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def load_model(layer, model):
    model.eval()

    features_names = layer  # the last conv layer of the resnet101
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    return model



#####################################################################
# settting
#####################################################################
sets = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = str(sets.gpu_id)
torch.manual_seed(sets.manual_seed)
print("Current setting is:")
print(sets)
print("\n\n")


#####################################################################
## getting backbone and hook the features
#####################################################################
sets.phase, sets.state = 'test', 2 # set to load model4cluster
backbone = generate_backbone_model(sets)
backbone = load_model(sets.target_layer, backbone)
backbone.eval()
print(backbone)


#####################################################################
# start
#####################################################################

Phase = ['train', 'test']
for phase in Phase:
    # Load Data
    root = sets.root_traindata if phase == 'train' else sets.root_testdata
    labels = {name: index for index in range(len(sets.classes)) for name in
              glob.glob(root + '/' + sets.modal + '/' + sets.classes[index] + '/' + phase + '/*.nii')}
    names = list(sorted(labels.keys()))


    #####################################################################
    # GradCAM/CAM visualization of a random subject
    #####################################################################
    random_img = random.choice(names)
    # random_img = 'C:/LYL_data/AD/ADNI/NC/train/*.nii'
    label = labels[random_img]
    ori_img, pre_img = Get_MRI_data(random_img)
    input_img = pre_img.unsqueeze(0).cuda()

    # forward pass, the result is the probability of [private, public]
    features_blobs = []
    _, logit = backbone.forward(input_img)
    h_x = F.softmax(logit.cpu(), 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    idx = idx.numpy()
    print('RESULT ON {}: {}'.format(random_img, idx[0]))

    # generate class activation mapping
    generate_camapping(backbone.parameters(), features_blobs, idx, ori_img, random_img, label, sets)


    #####################################################################
    ## get coordinate of peak response  (features for each channel)
    #####################################################################

    channel_features = []
    for i in range(sets.ccl_dim):  # corresponding to the channels
        channel_features.append([])

    for imgname in names:
        label = labels[imgname]
        ori_img, pre_img = Get_MRI_data(imgname)
        input_img = pre_img.unsqueeze(0).cuda()

        features_blobs = []
        logit = backbone.forward(input_img)

        for i, channel in enumerate(features_blobs[0]):

            tx, ty, tz = np.where(channel == channel.max())
            tx, ty, tz = tx[0], ty[0], tz[0]

            channel_features[i].append(tx)
            channel_features[i].append(ty)
            channel_features[i].append(tz)

    channel_features = np.array(channel_features)

    if not os.path.exists(sets.cluster_result_savepath):
        os.makedirs(sets.cluster_result_savepath)
    pd.DataFrame(channel_features).to_csv('{}/peak_result_{}_{}_{}.csv'
                 .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase))
    features_savename = '{}/peak_result_{}_{}_{}.npy' \
        .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase)
    np.save(features_savename, channel_features)

    print(channel_features.shape)  # channel_feature: [2048, 2 * sizeof_train_images]


    #####################################################################
    ## channel clustering
    #####################################################################
    # if sets.cluster_hierarchical is not True:
    #     # cluster one layer
    #     for part_num in sets.part_nums:
    #         part_index = KMeans(n_clusters=part_num, random_state=9).fit_predict(channel_features)
    #         cluster_savename = '{}/cluster_result_{}_{}_{}{}.npy' \
    #             .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase, str(part_num))
    #         np.save(cluster_savename, part_index)
    # else:
    #     # cluster more than one layer
    #     features = torch.from_numpy(channel_features).float()  # dim*(3*N)
    #     features = features.numpy()
    #     cluster_result = run_hkmeans(features, sets.part_nums_hierarchical)
    #
    #     # save the clustering result
    #     torch.save(cluster_result, os.path.join(sets.cluster_result_savepath, 'hcluster_result_{}_{}_{}{}'
    #               .format(sets.classes[0], sets.classes[1], phase, sets.part_nums_hierarchical)))
    #
    #     cluster_result_npy = np.array(cluster_result)
    #     cluster_result_txt = cluster_result
    #     features_savename = '{}/hcluster_result_{}_{}_{}{}.txt' \
    #         .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase, sets.part_nums_hierarchical)
    #     f = open(features_savename, 'w')
    #     for k, v in cluster_result_txt.items():
    #         f.write(str(k) + ':' + str(v) + ';\n')
    #     f.close

    # cluster one layer
    for part_num in sets.part_nums:
        part_index = KMeans(n_clusters=part_num, random_state=9).fit_predict(channel_features)
        cluster_savename = '{}/cluster_result_{}_{}_{}{}.npy' \
            .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase, str(part_num))
        np.save(cluster_savename, part_index)

    # cluster more than one layer
    features = torch.from_numpy(channel_features).float()  # dim*(3*N)
    features = features.numpy()
    cluster_result = run_hkmeans(features, sets.part_nums_hierarchical)

    # save the clustering result
    torch.save(cluster_result, os.path.join(sets.cluster_result_savepath, 'hcluster_result_{}_{}_{}{}.pth'
                                            .format(sets.classes[0], sets.classes[1], phase,
                                                    sets.part_nums_hierarchical)))

    cluster_result_npy = np.array(cluster_result)
    cluster_result_txt = cluster_result
    features_savename = '{}/hcluster_result_{}_{}_{}{}.txt' \
        .format(sets.cluster_result_savepath, sets.classes[0], sets.classes[1], phase, sets.part_nums_hierarchical)
    f = open(features_savename, 'w')
    for k, v in cluster_result_txt.items():
        f.write(str(k) + ':' + str(v) + ';\n')
    f.close










