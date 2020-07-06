import torch
import torch.nn.functional as F

from torch import nn

from utils.params import ParamDict as o

class FocalLoss(nn.Module):

    DEFAULT_PARAMS=o(
        alpha=.25,
        gamma=2,
        eps=1e-6,
    )

    def __init__(self, params=DEFAULT_PARAMS): # default paper settings
        super(FocalLoss, self).__init__()
        self.p = params
        self.alpha = nn.Parameter(torch.tensor(self.p.alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(self.p.gamma), requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(self.p.eps), requires_grad=False)

    def forward(self, output, target, mask, valid_channel_idx):
        # make sure output and targets are of same size
        if output.shape != target.shape:
            target = F.interpolate(target, (output.shape[2], output.shape[3]))
            mask = F.interpolate(mask, (output.shape[2], output.shape[3]))
        pt = output * target + (1 - output) * (1 - target)
        pt = torch.clamp(pt, self.eps, 1-self.eps)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -at * torch.pow(1-pt, self.gamma) * torch.log(pt)
        # Valid losses w.r.t. to annotated channels in different datasets
        valid_losses = loss[valid_channel_idx]
        # @see FCOS: nomalize loss wrt. number of positive (forground) pixels
        # Since there is no positive pixel in invalid channels, no need to mask target.
        norm = (target * mask).sum().clamp_min(1)
        return (valid_losses / norm).sum()
