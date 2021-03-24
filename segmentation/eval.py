import argparse
import os
import pathlib
import sys
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segmentation.loss import FocalLoss
from segmentation.model import FPNResNet18, NoFPNMobileNetV2Dilated
from utils.console import print_info, print_ok, print_warn
from utils.params import ParamDict

from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=str, required=True,
                        help='path to the parameter file')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='path to the trained model')
    return parser.parse_args()

def compute_acc(pred, label):
    '''
    pred: BHW
    label: BHW
    '''
    valid = (label >= 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def visualize_confusion_matrix(cmatrix, class_name_list):
    '''
    cmatrix: matrix of integer type (i.e., confusion count)
    '''
    assert len(cmatrix.shape) == 2
    cmatrix = cmatrix.astype(np.double)
    # normalize count to ratio
    for i in range(cmatrix.shape[0]):
        row_sum = np.sum(cmatrix[i])
        cmatrix[i] = cmatrix[i] / (row_sum + 1e-10)
    df_cm = pd.DataFrame(cmatrix, index = class_name_list, columns = class_name_list)
    plt.figure(figsize = (40,40))
    sn.heatmap(df_cm, annot=True)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    p = ParamDict.from_file(args.params)
    # data
    val_set = p.data.dataset(p.data.params, train=False)
    num_classes = p.model.num_classes
    # prediction labels
    label_range = [i for i in range(num_classes)]
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
    print_ok("Validation set has {} data points".format(len(val_set)))
    # get_class_names returns a dictionary of int-str pair
    # For validation, set batch size to 1.
    val_set = DataLoader(val_set, batch_size=1, shuffle=True,
                          pin_memory=True, num_workers=p.data.num_workers, drop_last=False)
    # model
    # late night
    model = NoFPNMobileNetV2Dilated(p.model)
    # resume / load pretrain if applicable
    assert args.model is not None, "Please specify path to trained model"
    weight_path = args.model
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict['model'])
    # transfer to GPU device
    device = torch.device('cuda:0')
    model.to(device)
    # Test
    model.eval()
    acc_meter = AverageMeter()
    for sample in val_set:
        image_b3hw = sample['image_b3hw'].to(device)
        seg_mask_bnhw = sample['seg_mask_bnhw'].to(device)
        valid_channel_idx_bc = sample['valid_label_idx'].to(device)
        with torch.no_grad():
            assert image_b3hw.shape[0] == 1, "during eval, only 1 image per batch!"
            output = model(image_b3hw)
            # Compute accuracy
            pred_map = output.max(dim = 1)[1]
            label_map = seg_mask_bnhw.max(dim = 1)[1]
            batch_acc, batch_fg_pixel_sum = compute_acc(pred_map, label_map)
            pred_map_np = pred_map.cpu().numpy()
            label_map_np = label_map.cpu().numpy()
            cur_confusion_matrix = confusion_matrix(label_map_np.reshape((-1,)), pred_map_np.reshape((-1,)), labels = label_range).astype(np.uint64)
            acc_meter.update(batch_acc, batch_fg_pixel_sum)
            total_confusion_matrix = total_confusion_matrix + cur_confusion_matrix

    print_info('[Testing] Overall Pixel Acc: {:.2f}%'.format(acc_meter.average() * 100))