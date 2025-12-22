import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- utils ----------
def compute_class_weights(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts[counts == 0] = 1
    return counts.sum() / (num_classes * counts)


# ---------- focal ----------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.weight is not None:
            focal_loss = self.weight[targets] * focal_loss

        return focal_loss.mean()


def get_loss(loss_name, labels=None, num_classes=None, gamma=2):
    if loss_name == "weighted_focal":
        return FocalLoss(weight=compute_class_weights(labels, num_classes), gamma=gamma)
    elif loss_name == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=compute_class_weights(labels, num_classes))
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
