import torch
import torch.nn.functional as F

# Focal Loss
def focal_loss(inputs, targets, alpha=0.7, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * ((1 - pt) ** gamma) * BCE_loss
    return F_loss.mean()

# Tversky Loss
def tversky_loss(logits, true_labels, alpha=0.5, beta=0.5, smooth=1.0):
    preds = torch.sigmoid(logits)
    true_pos = torch.sum(preds * true_labels)
    false_neg = torch.sum((1 - preds) * true_labels)
    false_pos = torch.sum(preds * (1 - true_labels))
    score = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - score

# Dice Loss
def dice_loss(logits, true_labels, smooth=1.0):
    preds = torch.sigmoid(logits)
    intersection = torch.sum(preds * true_labels, dim=0)
    union = torch.sum(preds, dim=0) + torch.sum(true_labels, dim=0)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score
