import torch
from torch import nn


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()  # 表示对所有的样本损失进行求和
    l2_lambda = 0.0001
    regularization_loss = 0
    for param in net.parameters():  # net.parameters()是一个生成器，它会返回列表中所有可训练的参数，param是模型中的参数，通常为tensor张量+
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


#used
class FocalLoss_v2(nn.Module):
    def __init__(self, num_class=2, gamma=2, alpha=None):

        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha == None:
            self.alpha = torch.ones(num_class)
        else:
            self.alpha=alpha

    def forward(self, logit, target):

        target = target.view(-1)

        alpha = self.alpha[target.cpu().long()]

        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()