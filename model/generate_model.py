import os
import torch
from torch import nn
from model.models_backbone import *



def generate_backbone_model(opt):
    assert opt.backbone in [
        'ConvMix_1',
        'Convmix_256', # for without_skull_image
        'Net_v2',
        'ConvMix_MRF_1',
        'ConvMix_MRF_CAG_1',
        'ConvMix_SWE_CWE_2',
        'ConvMix_MRF_CAG_SWE_CWE_2'
    ]

    if opt.backbone == 'ConvMix_1':
        # assert opt.backbone_depth in [10, 18, 34, 50, 101, 152, 200]
        backbone = ConvMix_1(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'Convmix_256':
        backbone = Convmix_256(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            kernel_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'Net_v2':
        backbone = Net_v2(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'ConvMix_MRF_1':
        backbone = ConvMix_MRF_1(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            convs_k=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'ConvMix_MRF_CAG_1':
        backbone = ConvMix_MRF_CAG_1(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            convs_k=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'ConvMix_SWE_CWE_2':
        backbone = ConvMix_SWE_CWE_2(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            n_classes=opt.n_cla_classes)
    elif opt.backbone == 'ConvMix_MRF_CAG_SWE_CWE_2':
        backbone = ConvMix_MRF_CAG_SWE_CWE_2(
            dim=opt.backbone_dim,
            patch_size=opt.backbone_patch,
            depth=opt.backbone_depth,
            conv_kernels=[3, 5, 7, 9],
            n_classes=opt.n_cla_classes)



    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

    # load pretrain for backbone pretraining
    if opt.state==1 and opt.model_backbone4pretrain:  # 0: backbone pretrain
        pretrain = torch.load(opt.model_backbone4pretrain)

        net_dict = backbone.state_dict()
        pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        backbone.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in backbone.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, backbone.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}
        backbone = backbone.cuda()
        return backbone, parameters

    elif opt.state==2 or opt.state==3 or opt.state==4:
        pretrain = torch.load(opt.model_backbone4cluster)

        if opt.state==3 or opt.state==4:
            backbone.avgp = nn.Sequential()
            backbone.flat = nn.Sequential()
            backbone.Linear = nn.Sequential()

        net_dict = backbone.state_dict()
        pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        backbone.load_state_dict(net_dict)

        backbone = backbone.cuda()
        return backbone

    return backbone, backbone.parameters()



# def generate_backbone_model(opt):
#     assert opt.backbone in [
#         'ConvMix_1',
#         'Convmix_256', # for without_skull_image
#         'Net_v2',
#         'ConvMix_MRF_1',
#         'ConvMix_MRF_CAG_1',
#         'ConvMix_SWE_CWE_2',
#         'ConvMix_MRF_CAG_SWE_CWE_2'
#     ]
#
#     if opt.backbone == 'ConvMix_1':
#         # assert opt.backbone_depth in [10, 18, 34, 50, 101, 152, 200]
#         backbone = ConvMix_1(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'Convmix_256':
#         backbone = Convmix_256(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             kernel_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'Net_v2':
#         backbone = Net_v2(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'ConvMix_MRF_1':
#         backbone = ConvMix_MRF_1(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             convs_k=[3, 5, 7, 9],
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'ConvMix_MRF_CAG_1':
#         backbone = ConvMix_MRF_CAG_1(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             convs_k=[3, 5, 7, 9],
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'ConvMix_SWE_CWE_2':
#         backbone = ConvMix_SWE_CWE_2(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             n_classes=opt.n_cla_classes)
#     elif opt.backbone == 'ConvMix_MRF_CAG_SWE_CWE_2':
#         backbone = ConvMix_MRF_CAG_SWE_CWE_2(
#             dim=opt.backbone_dim,
#             patch_size=opt.backbone_patch,
#             depth=opt.backbone_depth,
#             conv_kernels=[3, 5, 7, 9],
#             n_classes=opt.n_cla_classes)
#
#
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
#     # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     # backbone = backbone.cuda()
#     # net_dict = backbone.state_dict()
#
#
#     # load pretrain for backbone pretraining
#     if opt.phase != 'test' and opt.state==0:
#         backbone = backbone.cuda()
#         net_dict = backbone.state_dict()
#         print('loading pretrained backbone {}'.format(opt.model4backbone_pretrain))
#         pretrain = torch.load(opt.model4backbone_pretrain)
#         # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
#         pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
#
#         net_dict.update(pretrain_dict)
#         backbone.load_state_dict(net_dict)
#
#         new_parameters = []
#         for pname, p in backbone.named_parameters():
#             for layer_name in opt.new_layer_names:
#                 if pname.find(layer_name) >= 0:
#                     new_parameters.append(p)
#                     break
#
#         new_parameters_id = list(map(id, new_parameters))
#         base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, backbone.parameters()))
#         parameters = {'base_parameters': base_parameters,
#                       'new_parameters': new_parameters}
#
#         return backbone, parameters
#
#
#     # load pretrain for channel cluster or gcn
#     elif opt.phase == 'test':
#
#         pretrain = torch.load(opt.model4cluster)
#         print('loading pretrained backbone {}'.format(opt.model4cluster))
#
#         if opt.ccl_pretrain is True: # load backbone for channel cluster layer pretrain
#             backbone.avgp = nn.Sequential()
#             backbone.flat = nn.Sequential()
#             backbone.Linear = nn.Sequential()
#
#         # backbone.load_state_dict(torch.load(opt.model4cluster))
#
#         # pretrain = torch.load(opt.model4ccl_pretrain)
#         net_dict = backbone.state_dict()
#         pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
#         net_dict.update(pretrain_dict)
#         backbone.load_state_dict(net_dict)
#
#         backbone = backbone.cuda()
#
#         return backbone
#
#     return backbone, backbone.parameters()
#

