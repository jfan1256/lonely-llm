import torch
import torch.nn.functional as F

# Cross Entropy Loss
def ce_loss(inputs, targets, num_class):
    if num_class == 2:
        loss_ce_task = F.binary_cross_entropy_with_logits(inputs, targets)
        loss_ce = loss_ce_task
    elif num_class > 2:
        targets = targets.long()
        loss_ce_task = F.cross_entropy(inputs, targets)
        loss_ce = loss_ce_task
    return loss_ce

# Focal Loss
def focal_loss(inputs, targets, num_class, alpha=0.7, gamma=2.0):
    if num_class == 2:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * ((1 - pt) ** gamma) * BCE_loss
    elif num_class > 2:
        targets = targets.long()
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = alpha * ((1 - pt) ** gamma) * CE_loss
    return F_loss.mean()

# Tversky Loss
def tversky_loss(logits, true_labels, num_class, alpha=0.5, beta=0.5, smooth=1.0):
    if num_class == 2:
        preds = torch.sigmoid(logits)
        true_pos = torch.sum(preds * true_labels)
        false_neg = torch.sum((1 - preds) * true_labels)
        false_pos = torch.sum(preds * (1 - true_labels))
        score = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        return 1 - score
    elif num_class > 2:
        true_labels = true_labels.long()
        num_classes = logits.shape[1]
        preds = F.softmax(logits, dim=1)
        true_labels = F.one_hot(true_labels, num_classes=num_classes)
        true_pos = torch.sum(preds * true_labels, dim=0)
        false_neg = torch.sum((1 - preds) * true_labels, dim=0)
        false_pos = torch.sum(preds * (1 - true_labels), dim=0)
        score = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        return 1 - score.mean()

# Dice Loss
def dice_loss(logits, true_labels, num_class, smooth=1.0):
    if num_class == 2:
        preds = torch.sigmoid(logits)
        intersection = torch.sum(preds * true_labels, dim=0)
        union = torch.sum(preds, dim=0) + torch.sum(true_labels, dim=0)
        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score
    elif num_class > 2:
        true_labels = true_labels.long()
        num_classes = logits.shape[1]
        preds = F.softmax(logits, dim=1)
        true_labels = F.one_hot(true_labels, num_classes=num_classes)
        intersection = torch.sum(preds * true_labels, dim=0)
        union = torch.sum(preds, dim=0) + torch.sum(true_labels, dim=0)
        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score.mean()

# Center Loss
def center_loss(features, labels, centers, alpha):
    labels = labels.long()
    num_classes = centers.shape[0]
    centers_batch = centers[labels]
    diff = features - centers_batch
    squared_distances = torch.sum(torch.pow(diff, 2), dim=1)
    center_loss = torch.mean(squared_distances)
    unique_labels, unique_indices = torch.unique(labels, sorted=True, return_inverse=True)
    labels_count = torch.zeros(num_classes, device=features.device).index_add_(0, unique_labels, torch.ones_like(unique_labels, dtype=torch.float))
    accumulated_diffs = torch.zeros_like(centers).index_add_(0, labels, diff)
    if center_loss.requires_grad:
        def backward_hook(grad):
            centers.grad = -alpha * (accumulated_diffs / (labels_count.unsqueeze(1) + 1e-6))
            return grad
        center_loss.register_hook(backward_hook)
    return center_loss

# Large Margin Cosine (Analogous to Angular Loss)
def large_margin_cosine_loss(output, label, margin=0.35, scale=30.0):
    label = label.long()
    normalized_output = F.normalize(output, dim=1)
    cosine = torch.matmul(normalized_output, normalized_output.T)
    one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1)
    margin_cosine = cosine - one_hot * margin
    cosine = cosine - torch.eye(cosine.size(0), device=cosine.device) * cosine
    margin_cosine = margin_cosine - torch.eye(margin_cosine.size(0), device=margin_cosine.device) * margin_cosine
    output = torch.where(one_hot == 1, margin_cosine, cosine) * scale
    loss = F.cross_entropy(output, label)
    return loss

# Contrastive Loss for Encoder (Analogous to Euclidean Contrastive Loss)
def contrast_loss_encoder(features, labels, num_class, margin):
    if num_class == 2:
        embeddings_norm = F.normalize(features, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        match_loss = 0.5 * labels * (1 - cos_sim) ** 2
        non_match_loss = 0.5 * (1 - labels) * F.relu(margin - (1 - cos_sim)) ** 2
        return torch.mean(match_loss + non_match_loss)
    elif num_class > 2:
        embeddings_norm = F.normalize(features, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_eq = labels_eq.float()
        match_loss = 0.5 * labels_eq * (1 - cos_sim) ** 2
        non_match_loss = 0.5 * (1 - labels_eq) * F.relu(margin - (1 - cos_sim)) ** 2
        return torch.mean(match_loss + non_match_loss)

# Contrastive Loss for Encoder and Decoder (Analogous to Euclidean Contrastive Loss)
def contrast_loss_decoder(encoder_features, decoder_features, labels, margin):
    enc_norm = F.normalize(encoder_features, p=2, dim=1)
    dec_norm = F.normalize(decoder_features, p=2, dim=1)
    cos_sim = torch.mm(enc_norm, dec_norm.t())
    match_loss = 0.5 * labels * (1 - cos_sim) ** 2
    non_match_loss = 0.5 * (1 - labels) * F.relu(margin - (1 - cos_sim)) ** 2
    return torch.mean(match_loss + non_match_loss)

# Perplexity Loss
def perplex_loss(logits, targets):
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    perplexity = torch.exp(ce_loss)
    return perplexity

# Embedding Match Loss
def embed_match_loss(output_embeddings, target_embeddings):
    output_norm = F.normalize(output_embeddings, p=2, dim=1)
    target_norm = F.normalize(target_embeddings, p=2, dim=1)
    cos_sim = torch.sum(output_norm * target_norm, dim=1)
    return 1 - cos_sim.mean()