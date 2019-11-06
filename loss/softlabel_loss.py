import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """ Soft label loss.

    Parameter
    ---------
    classes : int
        number of classes
    smoothing : float
        amount to be smoothed. The label confidence = 1-smoothing
    weights : List[float]
        weights for different classes
    ignore : int
        one class label that won't cause smoothing

    """
    def __init__(self, classes, smoothing=0.0, weights=None, ignore=None, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weights = torch.Tensor(weights) if weights is not None else None
        self.cls = classes
        self.dim = dim
        self.ignore = ignore

    def forward(self, pred, target, compute_acc=False):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.to(pred.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore is not None:
                mask = (target.data.unsqueeze(1)==self.ignore).flatten()
                true_dist[mask] = 0.
                true_dist[mask, self.ignore] = 1.

        if self.weights is not None:
            losses = (torch.sum(-true_dist * pred * self.weights.to(pred.device), dim=self.dim))
        else:
            losses = (torch.sum(-true_dist * pred, dim=self.dim))

        if compute_acc:
            pred_label = pred.argmax(1)
            accs = (pred_label == target)
            return losses, accs
        else:
            return losses
