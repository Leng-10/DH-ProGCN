'''
Configs for training & testing
'''
import os
import argparse
import torch



def Data_Setting(traindataname, testdataname, modal, m, n):

    root = r'C:/LYL_data'
    shape = [91, 109, 91]  # size

    if testdataname == 'ADNI' or testdataname == 'AIBL':
        Classes = ['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        Class_size = [231, 81, 197, 108, 199]
        root = r'{}/AD'.format(root)

    elif testdataname == 'ABIDE':
        Classes = ['ABIDE-C', 'ABIDE-A']
        Class_size = [455, 413]

    elif testdataname == 'NIFD':
        Classes = ['NIFD-C', 'NIFD-N']
        Class_size = [92, 176]

    elif testdataname == 'PPMI':
        Classes = ['PPMI-HC', 'PPMI-PD-352']
        Class_size = [152, 449]

    elif testdataname == 'PSCI':
        Classes = ['non-PSCI', 'PSCI']
        Class_size = [80, 32]

    classes = [Classes[m], Classes[n]]


    trainroot = r'{}/{}'.format(root, traindataname)
    testroot = r'{}/{}'.format(root, testdataname)
    # if len(modal)==1:
    #     trainroot = r'{}/{}/{}'.format(root, traindataname, modal)
    #     testroot = r'{}/{}/{}'.format(root, testdataname, modal)
    # elif len(modal)==2:
    #     trainroot = r'{}/{}'.format(root, traindataname)
    #     testroot = r'{}/{}'.format(root, testdataname)

    sample_class_count = torch.Tensor([Class_size[m], Class_size[n]])
    class_weight = sample_class_count.float() / (Class_size[m] + Class_size[n])
    class_weight = 1. - class_weight

    return classes, class_weight, trainroot, testroot, shape



def parse_opts(
        traindataname='ADNI', # ADNI, PPMI, ABIDE, NIFD
        testdataname='ADNI',
        class1=0, #['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        class2=4,
        modal=['MRI'] # MRI, PET
        ):

    classes, class_weight, trainroot, testroot, shape = Data_Setting(
        traindataname, testdataname, modal, class1, class2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    # parser.add_argument('--gpu_id', nargs='+', type=int, help='Gpu id lists')
    parser.add_argument('--gpu_id', default='0', type=int, help='Gpu id for use')
    parser.add_argument('--n_cla_classes', default=2, type=int, help="Number of segmentation classes")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--n_epochs_backbone', default=500, type=int, help='Number of total epochs to run bakbone')
    parser.add_argument('--n_epochs_ccl', default=200, type=int, help='Number of total epochs to run channel cluster layer')
    parser.add_argument('--n_epochs_gcn', default=200, type=int, help='Number of total epochs to run gcn')
    parser.add_argument('--n_epoch_interval', default=20, type=int, help='Number of total epochs to run gcn')
    parser.add_argument('--saveacc', default=80, type=int, help='0-100')
    parser.add_argument('--saveauc', default=0.8, type=float, help='0-1')
    parser.add_argument('--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument('--state', default=3, type=int,
                        help='0: backbone pretrain; 1: cluster; 2: ccl pretrain; 3: backbone+gcn train')

    parser.add_argument('--modal', default=modal[0], type=str, help='MRI|PET')
    parser.add_argument('--classes', default=classes, type=list, help="['NC','SMC','sMCI','pMCI','AD']")
    parser.add_argument('--root_traindata', default=trainroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--root_testdata', default=testroot, type=str, help='Root directory path of loaddata')
    # parser.add_argument('--img_list', default='./loaddata/train.txt', type=str, help='Path for image list file')
    parser.add_argument('--input_D', default=shape[0], type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=shape[1], type=int, help='Input size of height')
    parser.add_argument('--input_W', default=shape[2], type=int, help='Input size of width')
    parser.add_argument('--w', default=class_weight, type=list)
    parser.add_argument('--weight_decay', default=1e-7, type=list)
    parser.add_argument('--lr_pretrain_backbone', default=0.0001, type=float,
                        help='Initial learning rate for backbone pretraining (divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune
    parser.add_argument('--lr_pretrain_ccl', default=0.0001, type=float,
                        help='Initial learning rate for channel cluster layer pretraining (divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='Initial learning rate for backbone (divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune
    parser.add_argument('--lr_ccl', default=1e-3, type=float,
                        help='Initial learning rate for ccl (divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune
    parser.add_argument('--lr_gcn', default=1e-4, type=float,
                        help='Initial learning rate for gcn(divided by 10 while training by lr scheduler)')  # set to 0.001 when finetune

    # parser.add_argument('--pretrain_path', default='./trails/pretrain/resnet_18.pth', type=str, help='Path for pretrained model.') # default='pretrain/resnet_34.pth',
    parser.add_argument('--backbone', default='Net_v2', type=str, help='(ConvMix_1 | Convmix_256 | Net_v2 | ConvMix_MRF_1 | ConvMix_MRF_CAG_1 ')
    parser.add_argument('--backbone_patch', default=5, type=int, help='Depth of net (3 | 5 | 7 | 9 | 11)')
    parser.add_argument('--backbone_depth', default=5, type=int, help='Depth of net (5 | 7 | 9 | 11)')
    parser.add_argument('--backbone_dim', default=1024, type=int)

    parser.add_argument('--cluster_hierarchical', default=False, help=' model for channel cluster')
    parser.add_argument('--part_nums', default=list(range(2, 14, 2)), help=' model for channel cluster')
    parser.add_argument('--part_num', default=8, help=' model for channel cluster')
    parser.add_argument('--part_nums_hierarchical', default=[16, 8, 4], help=' model for channel cluster')
    parser.add_argument('--T', default=0.2, type=float, help='Tempreture')
    parser.add_argument('--ccl_dim', default=512, type=int)

    parser.add_argument('--model_backbone4pretrain', default='./trails/model_pretrain_backbone/ADNI-MRI NC vs. AD/Net_v2(5, 5, 1024) w=tensor([0.4628, 0.5372])/'
                                'epoch84_acc0.87_auc79.6_sen0.8_spe0.79_NC_AD.pth', type=str, help='Path for pretrained backbone.')  # default='pretrain/resnet_34.pth',
    parser.add_argument('--new_layer_names', default='', help='change the last layer for another ')
    parser.add_argument('--target_layer', default=['conv22', 'avgp'], help='feature of layers for channel cluster')
    # parser.add_argument('--model4cluster', default='', help = 'backbone pretrained model for channel cluster')
    parser.add_argument('--model_backbone4cluster', default='./trails/model_pretrain_backbone/ADNI-MRI NC vs. AD/Net_v2(5, 5, 1024) w=tensor([0.4628, 0.5372])/'
                                'epoch84_acc0.87_auc79.6_sen0.8_spe0.79_NC_AD.pth', help='backbone pretrained model for channel cluster')
    parser.add_argument('--model_ccl4pretrain', default='', help='Path for pretrained channel cluster layer.')
    parser.add_argument('--model_ccl', default='./trails/model_pretrain_ccl/ADNI-MRI NC vs. AD/Net_v2(5, 5, 1024) w=tensor([0.4628, 0.5372])'
                                               '/8/channel_cluster_layer(41)_0.31331086291207205.pth', help='Path for pretrained channel cluster layer.')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')

    # parser.add_argument('--save_path', default=r'./trails/model_{}/{} {} vs {}/{}({},{}) w={}'
    #                     .format(modal, testdataname, classes[0], classes[1],args.model, ), type=str, help='Path for resume model.')

    args = parser.parse_args()

    args.pretrain_savepath_backbone = r'./trails/model_pretrain_backbone/{}-{}_{}vs.{}/{}({},{},{})_w={}'\
        .format(testdataname, args.modal, classes[0], classes[1], args.backbone,
                args.backbone_patch, args.backbone_depth, args.backbone_dim, args.w)
    args.pretrain_savepath_ccl = r'./trails/model_pretrain_ccl/{}-{}_{}vs.{}/{}({},{},{})_w={}/' \
        .format(testdataname, args.modal, classes[0], classes[1], args.backbone,
                args.backbone_patch, args.backbone_depth, args.backbone_dim, args.w)
    args.cluster_result_savepath = r'./trails/cluster_result/{}-{}_{}vs.{}/{}({},{},{})_{}' \
        .format(testdataname, args.modal, classes[0], classes[1], args.backbone,
                args.backbone_patch, args.backbone_depth, args.backbone_dim, os.path.split(args.model_backbone4cluster)[1])
    args.gradcam_savepath = r'./trails/gradcam/{}-{}_{}vs.{}' \
        .format(testdataname, args.modal, classes[0], classes[1])
    args.sum_savpath = r'./trails/models/{}-{}_{}vs.{}'.format(testdataname, args.modal, classes[0], classes[1])

    args.log_savpath = r'./log/{}-{}_{}vs.{}'.format(testdataname, args.modal, classes[0], classes[1])

    return args









def parse_opts_single(
        traindataname='ADNI', # ADNI, PPMI, ABIDE, NIFD
        testdataname='ADNI',
        class1=0, #['NC', 'SMC', 'sMCI', 'pMCI', 'AD']
        class2=4,
        modal=['MRI'] # MRI, PET
        ):

    classes, class_weight, trainroot, testroot, shape = Data_Setting(
        traindataname, testdataname, modal, class1, class2)

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_cla_classes', default=2, type=int, help="Number of segmentation classes")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')# set to 0.001 when finetune
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch Size')
    parser.add_argument('--phase', default='train', type=str, help='Phase of train or test')
    # parser.add_argument('--save_intervals', default=50, type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=500, type=int, help='Number of total epochs to run')
    parser.add_argument('--saveacc', default=55, type=int, help='0-100')
    parser.add_argument('--saveauc', default=0.5, type=float, help='0-1')

    parser.add_argument('--modal', default=modal[0], type=str, help='MRI|PET')
    parser.add_argument('--classes', default=classes, type=list, help="['NC','SMC','sMCI','pMCI','AD']")
    parser.add_argument('--root_traindata', default=trainroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--root_testdata', default=testroot, type=str, help='Root directory path of loaddata')
    parser.add_argument('--w', default=class_weight, type=list)
    # parser.add_argument('--img_list', default='./loaddata/train.txt', type=str, help='Path for image list file')

    parser.add_argument('--input_D', default=shape[0], type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=shape[1], type=int, help='Input size of height')
    parser.add_argument('--input_W', default=shape[2], type=int, help='Input size of width')

    parser.add_argument('--pretrain_path', default=False, type=str,help='Path for pretrained model.')  # default='pretrain/resnet_34.pth',
    # parser.add_argument('--pretrain_path', default='./trails/pretrain/resnet_18.pth', type=str, help='Path for pretrained model.') # default='pretrain/resnet_34.pth',
    # parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='New layer except for backbone')#default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
    parser.add_argument('--gpu_id', nargs='+', type=int, help='Gpu id lists')
    parser.add_argument('--model', default='Net_v2', type=str, help='(ConvMix_1 | Convmix_256 | Net_v2 | ConvMix_MRF_1 | ConvMix_MRF_CAG_1 ')
    parser.add_argument('--model_patch', default=5, type=int, help='Depth of net (3 | 5 | 7 | 9 | 11)')
    parser.add_argument('--model_depth', default=5, type=int, help='Depth of net (5 | 7 | 9 | 11)')
    parser.add_argument('--model_dim', default=1024, type=int)

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')

    # parser.add_argument('--save_path', default=r'./trails/model_{}/{} {} vs {}/{}({},{}) w={}'
    #                     .format(modal, testdataname, classes[0], classes[1],args.model, ), type=str, help='Path for resume model.')

    args = parser.parse_args()

    args.save_path = r'./trails/model_{}/{} {}/{}({},{}) w={}'\
        .format(args.modal,
                testdataname,  classes,
                args.model, args.model_patch, args.model_depth, args.w)

    return args


