import xlwt
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, auc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_setting(args):
    argsDict = args.__dict__
    with open('setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def create_result_xls(save_path, classes, args):
    file = xlwt.Workbook(encoding='utf-8')
    sheet = file.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet.write_merge(0, 0, 2, 7, 'train')
    sheet.write_merge(0, 0, 8, 14, 'test')
    sheet.write(1, 0, 'Epoch')
    sheet.write(1, 1, 'lr')
    # sheet.write(1, 1, 'loss')
    col = ('loss', 'ACC', 'AUC', 'F1', 'SEN', 'SPE')
    for i in range(0, 6):
        sheet.write(1, i + 2, col[i])
        sheet.write(1, i + 8, col[i])

    sheet_sets = file.add_sheet('sets', cell_overwrite_ok=True)
    argsDict = args.__dict__
    i = 0
    for eachArg, value in argsDict.items():
        if type(value)==torch.Tensor:
            value = value.tolist()
            value = [str(x) for x in value]
        if type(value)==list and type(value[0])==int:
            value = [str(x) for x in value]

        sheet_sets.write(i, 0, eachArg)
        sheet_sets.write(i, 1, value)
        i = i + 1


    file_name = '{}/{}_{} {}.xls'.format(save_path, classes[0], classes[1], time.strftime('%m-%d %H-%M', time.localtime(time.time())))
    file.save(file_name)
    return file, file_name, sheet


def index_to_label(part_indexs, part_num):
    cluster_label = []
    for i in range(part_num):
        cluster_label.append([])

    for index in part_indexs:
        for j in range(part_num):
            cluster_label[j].append(0)
        cluster_label[index][-1] = 1

    return cluster_label


def calculate(label, predicted, predicted1):
    TP, FP, FN, TN = 0, 0, 0, 0

    for i, x in enumerate(label):
        # TP    predict 和 label 同时为1
        TP += ((predicted[i] == 1) & (label[i] == 1)).sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted[i] == 0) & (label[i] == 0)).sum()
        # FN    predict 0 label 1
        FN += ((predicted[i] == 0) & (label[i] == 1)).sum()
        # FP    predict 1 label 0
        FP += ((predicted[i] == 1) & (label[i] == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    # p, r = 0, 0
    spe = TN / (TN + FP)
    sen = TP / (TP + FN)
    acc = 100. * (TP + TN) / (TP + TN + FP + FN)
    auc = roc_auc_score(label, predicted1)
    ave = average_precision_score(label, predicted1)
    F1_score = f1_score(label, predicted)

    return acc, auc, F1_score, sen, spe, ave


def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  # ##计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # ##计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  # ##假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()