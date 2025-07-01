import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    for x, pos, hf, y in data_iter:
        x, pos, hf, y = x.cuda(), pos.cuda(), hf.cuda(), y.cuda()
        outputs, rep= net(x, pos, hf)  # output为最后分类的（15，2）维张量，rep为block1输出后的（15，128）维张量
        pred_prob_positive = outputs[:, 1]  # 选择output输出的第二列
        pred_prob = pred_prob + pred_prob_positive.tolist()  # 把每个批次的第二列保存在pred_prob列表中
        label_pred = label_pred + outputs.argmax(dim=1).tolist()  # 在output列维度中取出最大元素的索引作为预测标签
        label_real = label_real + y.tolist()  # 将真实标签添加到label_real中，+表示列表的拼接，.tolist将tensor张量转换为python列表
        rep_list.extend(rep.detach().cpu().numpy())  # rep为张量，detach()将rep 从计算图中分离，使其不再跟踪梯度，然后转换为numpy数组
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, rep_list, label_real


# # 评价指标

def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)  # 真实标签长度
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / ((tp + fn) + tn + fp)
    # ACC = float(tp + tn) /len(label_real)
    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC，FPR代表真正率，TPR代表假阳性（实为负，判为正）
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data
