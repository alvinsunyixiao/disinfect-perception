import argparse
import os
import pathlib
import time
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from segmentation.data import COCODataset
from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18
from utils.params import ParamDict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, required=True,
                        help='path to the parameter file')
    parser.add_argument('--logdir', type=str, required=True,
                        help='directories to store logging and checkpoints output')
    parser.add_argument('--tag', type=str, default=None,
                        help='optional tag name appended to the session name')
    return parser.parse_args()

def get_session_dir(logdir, tag):
    session_name = time.strftime("sess_%y-%m-%d_%H-%M-%S")
    if tag is not None:
        session_name += '_' + tag
    session_dir = os.path.join(logdir, session_name)
    # create if not existed
    pathlib.Path(session_dir).mkdir(parents=True, exist_ok=True)
    return session_dir

if __name__ == '__main__':
    args = parse_arguments()
    p = ParamDict.from_file(args.params)
    sess_dir = get_session_dir(args.logdir, args.tag)
    # data
    coco_train = COCODataset(p.data)
    dataloader = DataLoader(coco_train, batch_size=p.data.batch_size, shuffle=True,
                            pin_memory=True, num_workers=p.data.num_workers, drop_last=True)
    # model
    model = FPNResNet18(p.model)
    fl = FocalLoss(p.loss)
    # optimizer
    # exclude weight decay from batch norms
    bn_params = []
    non_bn_params = []
    for name, param in model.named_parameters():
        if 'bn' in name:
            bn_params.append(param)
            print('Found Batch Norm Param: {}'.format(name))
        else:
            non_bn_params.append(param)
    optimizer = torch.optim.SGD([
        {'params': bn_params, 'weight_decay': 0},
        {'params': non_bn_params},
    ], lr=p.trainer.lr_init, momentum=p.trainer.lr_momentum, weight_decay=p.trainer.weight_decay)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, p.trainer.lr_schedule)
    # logging
    writer = SummaryWriter(sess_dir)
    image_b3hw = torch.zeros((1,3,256,256))
    writer.add_graph(model, image_b3hw)
    # transfer to GPU device
    device = torch.device('cuda:0')
    model.to(device)
    fl.to(device)
    # training loop
    for epoch in range(100):
        # log learning rate
        writer.add_scalar('lr', lr_schedule.get_last_lr()[0], epoch * len(dataloader))
        # per-epoch running average loss
        running_loss = 0
        for i, sample in enumerate(dataloader):
            image_b3hw = sample['image_b3hw'].to(device)
            seg_mask_bnhw = sample['seg_mask_bnhw'].to(device)
            loss_mask_b1hw = sample['loss_mask_b1hw'].to(device)
            # prevent accumulation
            optimizer.zero_grad()
            # forward inference
            output = model(image_b3hw)
            loss = 0
            for o in output:
                loss += fl(o, seg_mask_bnhw, loss_mask_b1hw)
            # backward inference
            loss.backward()
            # gradient update
            optimizer.step()
            # loss logging
            running_loss += loss.item()
            print('Epoch: %d Loss: %.4f' % (epoch, running_loss / (i + 1)))
            if i % 50 == 0:
                writer.add_scalar('loss', loss.item(), epoch * len(dataloader) + i)
        # update learning rate
        lr_schedule.step()
        # save checkpoints
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_shedule': lr_schedule.state_dict(),
            'epoch': epoch,
        }, os.path.join(sess_dir, 'epoch-{}.pth'.format(epoch)))

