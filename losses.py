import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        targets = targets.view(-1, 1)
        logpt = F.log_softmax(logits, dim=1)
        logpt = logpt.gather(1, targets).view(-1)
        pt = logpt.exp()
        if self.weight is not None:
            if self.weight.type() != logits.data.type():
                self.weight = self.weight.type_as(logits.data)
            weight = self.weight.gather(0, targets.data.view(-1))
            logpt = logpt * weight
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

class MixLoss(nn.Module):
    def __init__(self, weight=None, args=None):
        super(MixLoss, self).__init__()
        self.weight = weight
        self.args = args
        self.args.cls_num_list = torch.tensor(self.args.cls_num_list, requires_grad=False).to(args.gpu)

    def forward(self, logit, target):
        n_max = max(self.args.cls_num_list)
        phi = self.args.temp_epsilon * len(self.args.cls_num_list) * (self.args.cls_num_list / n_max) + (1 - self.args.temp_epsilon)
        phi = max(phi)/phi
        phi = phi**(1/self.args.temp_eta)
        logit = logit/phi
        pred = torch.softmax(logit, dim=1)

        if self.weight is not None:
            return - torch.mean(torch.sum(torch.log(pred) * target * self.weight, dim=1) + torch.sum(torch.log(1-pred) * (1-target)* self.weight, dim=1))
        else:
            return - torch.mean(torch.sum(torch.log(pred) * target, dim=1) + torch.sum(torch.log(1-pred) * (1-target), dim=1))


def mixup(input, target, input_b, target_b, args):
    if args.mixup_beta > 0:
        lam = np.random.beta(args.mixup_beta, args.mixup_beta)
    else:
        lam = 1
    if args.use_experts:
        lam = max(1-lam, lam)

    batch_size = input.shape[0]
    index = torch.randperm(batch_size)
    target = F.one_hot(target, args.num_classes)
    target_b = F.one_hot(target_b, args.num_classes)
    
    if args.use_experts:
        mixed_input = lam * input[index] + (1 - lam) * input_b
        mixed_target = lam * target[index] + (1 - lam) * target_b
        mixed_input_2 = (1 - lam) * input[index] + lam * input_b
        mixed_target_2 = (1 - lam) * target[index] + lam * target_b
        return mixed_input, mixed_input_2, mixed_target, mixed_target_2
    else:
        mixed_input = lam * input[index] + (1 - lam) * input_b
        mixed_target = lam * target[index] + (1 - lam) * target_b
    return mixed_input, mixed_target

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
