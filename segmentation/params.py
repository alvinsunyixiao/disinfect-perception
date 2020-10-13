import torch

from data.data import COCODataset, FineGrainedADE20KDataset, HospitalDataset, DatasetMixer
from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18
from utils.params import ParamDict as o

data = o(
    batch_size=32,
    num_workers=12,
    dataset=DatasetMixer,
    params=(
        (FineGrainedADE20KDataset, FineGrainedADE20KDataset.DEFAULT_PARAMS), 
        (HospitalDataset, HospitalDataset.DEFAULT_PARAMS)
    ),
)

def lr_schedule(epoch):
    if epoch < 40:
        return 1e-0
    elif epoch < 70:
        return 1e-1
    else:
        return 1e-2

trainer=o(
    lr_init=1e-2,
    lr_momentum=0.9,
    lr_schedule=lr_schedule,
    weight_decay=1e-4,
    mixed_precision=True,
    num_epochs=100,
)

PARAMS=o(
    data=data,
    loss=FocalLoss.DEFAULT_PARAMS,
    model=FPNResNet18.DEFAULT_PARAMS,
    trainer=trainer,
)

def resolve_dependancies(params):
    params.model.update(num_classes=24)

resolve_dependancies(PARAMS)

