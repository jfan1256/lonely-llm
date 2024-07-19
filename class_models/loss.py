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

# Center Loss
def center_loss(features, labels, centers, alpha=0.5):
    labels = labels.long()
    centers_batch = centers[labels]
    diff = features - centers_batch
    updated_centers = centers_batch - alpha * diff
    centers.data[labels] = updated_centers.data
    loss = (diff ** 2).sum() / (2.0 * features.shape[1])
    return loss

# Angular Loss (Analogous to Triplet Loss)
def angular_loss(features, labels, margin=45, eps=1e-8):
    dot = torch.matmul(features, features.t())
    norms = features.norm(p=2, dim=1)
    cosine = dot / (norms[:, None] * norms[None, :] + eps)
    angles = torch.acos(torch.clamp(cosine, -1 + eps, 1 - eps)) * 180 / torch.pi
    # Create masks for positive (same class) and negative (different class) pairs
    positive_mask = labels[:, None] == labels[None, :]
    negative_mask = labels[:, None] != labels[None, :]
    # Angular loss calculation
    positive_loss = torch.mean(F.relu(margin - angles[positive_mask]))
    negative_loss = torch.mean(F.relu(angles[negative_mask] - margin))
    return positive_loss + negative_loss

# Constrastive Loss (Analogous to Euclidean Constrastive Loss)
def constrast_loss(features, labels, margin):
    embeddings_norm = F.normalize(features, p=2, dim=1)
    cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
    match_loss = 0.5 * labels * (1 - cos_sim) ** 2
    non_match_loss = 0.5 * (1 - labels) * F.relu(margin - (1 - cos_sim)) ** 2
    return torch.mean(match_loss + non_match_loss)
