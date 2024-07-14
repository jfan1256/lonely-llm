import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha, gamma):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * ((1 - pt) ** gamma) * BCE_loss
    return F_loss.mean()