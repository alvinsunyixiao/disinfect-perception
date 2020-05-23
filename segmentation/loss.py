import torch
import torch.nn.functional as F

from torch import nn

class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2, eps=1e-6): # default paper settings
        super(FocalLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)

    def forward(self, output, target, inv_mask):
        # make sure output and targets are of same size
        if output.shape != target.shape:
            target = F.interpolate(target, (output.shape[2], output.shape[3]))
            inv_mask = F.interpolate(inv_mask, (output.shape[2], output.shape[3]))
        pt = output * target + (1 - output) * (1 - target)
        pt = torch.clamp(pt, self.eps, 1-self.eps)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -at * torch.pow(1-pt, self.gamma) * torch.log(pt)
        val_mask = (1 - inv_mask)
        loss *= val_mask / val_mask.sum()
        return loss.sum()
