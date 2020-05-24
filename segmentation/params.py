import torch

from segmentation.data import COCODataset
from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18
from utils.params import ParamDict as o

data = COCODataset.DEFAULT_PARAMS(
    batch_size=32,
    num_workers=10,
)

def lr_schedule(epoch):
    if epoch < 40:
        return 1e0
    elif epoch < 80:
        return 1e-1
    else:
        return 1e-2

trainer=o(
    lr_init=1e-2,
    lr_momentum=0.9,
    lr_schedule=lr_schedule,
    weight_decay=1e-4,
)

PARAMS=o(
    data=data,
    loss=FocalLoss.DEFAULT_PARAMS,
    model=FPNResNet18.DEFAULT_PARAMS,
    trainer=trainer,
)

def resolve_dependancies(params):
    params.model.update(num_classes=len(params.data.classes))

resolve_dependancies(PARAMS)

