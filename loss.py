import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, weight, alpha, beta):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight
        self.weight_tversky = 1 - weight
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss_ce = self.ce_loss(inputs, targets)

        loss_tversky = tversky_loss(targets, inputs, self.alpha, self.beta)

        combined_loss = self.weight_ce * loss_ce + self.weight_tversky * loss_tversky
        
        return combined_loss

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    num_classes = logits.shape[1]
    true_1_hot = F.one_hot(true, num_classes=num_classes).float()

    # Apply softmax to logits to get probabilities
    probas = F.softmax(logits, dim=1)

    # Compute the Tversky loss
    intersection = torch.sum(probas * true_1_hot, dim=0)
    fps = torch.sum(probas * (1 - true_1_hot), dim=0)
    fns = torch.sum((1 - probas) * true_1_hot, dim=0)

    num = intersection
    denom = intersection + alpha * fps + beta * fns

    tversky_index = (num / (denom + eps)).mean()
    return 1 - tversky_index