import torch
from torch import Tensor, nn
import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma=0, **kwargs):
        kwargs.update(reduction='none')
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma

    def forward(self, logits: Tensor, labels:Tensor):
        ce = super(FocalLoss, self).forward(logits, labels)
        pt = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1).detach()
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()
class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Unsupported reduction: {reduction}')
        self.reduction = reduction
    def forward(self, pred_mean, pred_logvar, target):
        """Guassian negative log-likelihood loss."""
        var = torch.exp(pred_logvar).clamp(min=1e-6)
        loss =  0.5 * (pred_logvar + (target - pred_mean) ** 2 / var)
        if self.reduction != "none":
            return getattr(loss, self.reduction)()
        else:
            return loss
