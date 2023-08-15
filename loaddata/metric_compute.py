from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, roc_curve, auc
import matplotlib.pyplot as plt


def getMCA(correct, predicted, predicted1):
    acc = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i, x in enumerate(correct):
        # TP    predict 和 label 同时为1
        TP += ((predicted[i] == 1) & (correct[i] == 1)).sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted[i] == 0) & (correct[i] == 0)).sum()
        # FN    predict 0 label 1
        FN += ((predicted[i] == 0) & (correct[i] == 1)).sum()
        # FP    predict 1 label 0
        FP += ((predicted[i] == 1) & (correct[i] == 0)).sum()


    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = acc * 100
    auc = roc_auc_score(correct, predicted1)
    F1_score = f1_score(correct, predicted)
    spe = TN / (TN + FP)
    sen = TP / (TP + FN)

    return acc, auc, F1_score, sen, spe


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