import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from segmentation.data import COCODataset
from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18

if __name__ == '__main__':
    # data
    coco_train = COCODataset()
    dataloader = DataLoader(coco_train, batch_size=32, shuffle=True,
                            pin_memory=True, num_workers=16, drop_last=True)
    # model
    model = FPNResNet18(128, len(coco_train.p.classes))
    fl = FocalLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-2, momentum=0.9, weight_decay=1e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    # logging
    writer = SummaryWriter('/home/alvin/checkpoints/baseline')
    image_b3hw = torch.zeros((1,3,256,256))
    writer.add_graph(model, image_b3hw)
    # transfer to GPU device
    device = torch.device('cuda:0')
    model.to(device)
    fl.to(device)
    # training loop
    for epoch in range(100):
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

        lr_schedule.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_shedule': lr_schedule.state_dict(),
            'epoch': epoch,
        }, '/home/alvin/checkpoints/epoch-{}.pth'.format(epoch))

