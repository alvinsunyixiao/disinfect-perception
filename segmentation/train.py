import argparse
import os
import pathlib
import sys
import time
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18
from utils.console import print_info, print_ok, print_warn
from utils.params import ParamDict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, required=True,
                        help='path to the parameter file')
    parser.add_argument('--logdir', type=str, required=True,
                        help='directories to store logging and checkpoints output')
    parser.add_argument('--tag', type=str, default=None,
                        help='optional tag name appended to the session name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to the checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='path to a saved weight checkpoint to initialize backbone')
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
    # data
    train_set = p.data.dataset(p.data.params, train=True)
    val_set = p.data.dataset(p.data.params, train=False)
    print_ok("Training set has {} data points.".format(len(train_set)))
    print_ok("Validation set has {} data points".format(len(val_set)))
    train_set = DataLoader(train_set, batch_size=p.data.batch_size, shuffle=True,
                            pin_memory=True, num_workers=p.data.num_workers, drop_last=True)
    val_set = DataLoader(val_set, batch_size=p.data.batch_size, shuffle=True,
                          pin_memory=True, num_workers=p.data.num_workers, drop_last=False)
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
            print_warn('Found Batch Norm Param: {}'.format(name))
        else:
            non_bn_params.append(param)
    optimizer = torch.optim.SGD([
        {'params': bn_params, 'weight_decay': 0},
        {'params': non_bn_params},
    ], lr=p.trainer.lr_init, momentum=p.trainer.lr_momentum, weight_decay=p.trainer.weight_decay)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, p.trainer.lr_schedule)
    # resume / load pretrain if applicable
    epoch_start = 0
    if args.resume is not None or args.pretrained is not None:
        assert (args.pretrained is None) ^ (args.resume is None),\
            "Trainer resuming and using pretrained are mutually exclusive!"
        weight_path = args.resume or args.pretrained
        state_dict = torch.load(weight_path)
        if args.pretrained is not None:
            # Load pretrained weights
            pretrained_weight_dict = state_dict
            if 'model' in pretrained_weight_dict:
                pretrained_weight_dict = pretrained_weight_dict['model']
            model.backbone.load_state_dict(pretrained_weight_dict, strict=False)
        else:
            # Resume unfinished training
            epoch_start = state_dict['epoch'] + 1
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            lr_schedule.load_state_dict(state_dict['lr_schedule'])
    # logging
    sess_dir = get_session_dir(args.logdir, args.tag)
    train_writer = SummaryWriter(os.path.join(sess_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(sess_dir, 'val'))
    if isinstance(p.data.params, tuple):
        # Handle multiple dataset inputs
        image_b3hw = torch.zeros((1,3) + p.data.params[0][1].crop_params.output_hw)
    else:
        image_b3hw = torch.zeros((1,3) + p.data.params.crop_params.output_hw)
    # TODO: place this param in params.py
    accumulation_step = 4
    train_writer.add_graph(model, image_b3hw)
    # transfer to GPU device
    device = torch.device('cuda:0')
    model.to(device)
    fl.to(device)
    # mixed precision preparation
    if p.trainer.mixed_precision:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    # training loop
    for epoch in range(epoch_start, p.trainer.num_epochs):
        # log learning rate
        train_writer.add_scalar('lr', lr_schedule.get_last_lr()[0], epoch * len(train_set))
        # TRAIN
        model.train(True)
        running_loss = 0
        optimizer.zero_grad()
        with tqdm(train_set, dynamic_ncols=True) as t:
            for i, sample in enumerate(t):
                image_b3hw = sample['image_b3hw'].to(device)
                seg_mask_bnhw = sample['seg_mask_bnhw'].to(device)
                loss_mask_bnhw = sample['loss_mask_bnhw'].to(device)
                valid_channel_idx_bc = sample['valid_label_idx'].to(device)
                # forward inference
                if p.trainer.mixed_precision:
                    with autocast(True):
                        output = model(image_b3hw)
                        loss = fl(output[-1], seg_mask_bnhw, loss_mask_bnhw, valid_channel_idx_bc)
                else:
                    output = model(image_b3hw)
                    loss = fl(output[-1], seg_mask_bnhw, loss_mask_bnhw, valid_channel_idx_bc)
                # backward optimize
                if p.trainer.mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % accumulation_step == 0:
                    if p.trainer.mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                # loss logging
                running_loss += loss.item()
                t.set_description_str('[EPOCH %d] Loss: %.4f' %
                                      (epoch, running_loss / (i+1)))
                if i % 50 == 0:
                    train_writer.add_scalar('loss', loss.item(), epoch * len(train_set) + i)
        # VALIDATION
        model.eval()
        running_loss = 0
        for sample in val_set:
            image_b3hw = sample['image_b3hw'].to(device)
            seg_mask_bnhw = sample['seg_mask_bnhw'].to(device)
            loss_mask_bnhw = sample['loss_mask_bnhw'].to(device)
            valid_channel_idx_bc = sample['valid_label_idx'].to(device)
            with torch.no_grad():
                output = model(image_b3hw)
                # TODO: add back output[-1]
                loss = fl(output, seg_mask_bnhw, loss_mask_bnhw, valid_channel_idx_bc)
                running_loss += loss.item()
        val_loss = running_loss / len(val_set)
        running_loss = 0
        val_writer.add_scalar('loss', val_loss, (epoch+1) * len(train_set))
        print_info('Validation Loss: %.4f' % val_loss)
        # update learning rate
        lr_schedule.step()
        # save checkpoints
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': lr_schedule.state_dict(),
            'epoch': epoch,
        }, os.path.join(sess_dir, 'epoch-{}-{:.4f}.pth'.format(epoch, val_loss)))

