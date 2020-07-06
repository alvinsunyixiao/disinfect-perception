import os
import numpy as np
import torch

from segmentation.label_params import\
    final_classes, coco_label_map, ade20k_label_map, hospital_label_map

# Mapping scheme:
#   encoded number -> native class name -> mapped class name -> mapped class index

# Assume that 0 represents background so that labelling starts from 1

def get_label_unifier(raw_label_dict, string_mapping_dict, final_name_list):
    '''
    Input
        - raw_label_dict: an integer-string dictionary that maps raw seg mask to meaningful strings
        - string_mapping_dict: a string-string dictionary such that it maps input name
                                to output name in final label classes
        - final_name_list: a list of ordered strings that represent classes in
                            final set up
    Return
        - label_map_func: a vectorized function which maps input segmentation mask
                            to output mask (see the corresponding function factory
                            for details)
        - valid_class_list: a list of integers that start from 0 to len(final_name_list) - 1,
                            which is intended to be directly used to mask loss during BP.
    '''
    output_class_num = len(final_name_list)
    valid_class_list = set()
    mapping_func = {}
    for input_idx in raw_label_dict.keys():
        input_name = raw_label_dict[input_idx]
        assert isinstance(input_idx, int)
        assert isinstance(input_name, str)
        if input_name not in string_mapping_dict:
            continue # not relevant classes are mapped to zero
        output_name = string_mapping_dict[input_name]
        output_idx = final_name_list.index(output_name) + 1
        valid_class_list.add(output_idx - 1) # from 1-based to 0-based
        mapping_func[input_idx] = output_idx
    # Function factory
    def label_map_func(label_mat):
        '''
        Input:
            - label_mat: a 2D torch tensor of shape (H, W) and of type integer
                which encodes class indices.
        '''
        ret_mat = torch.zeros_like(label_mat)
        for input_idx in mapping_func:
            output_idx = mapping_func[input_idx]
            ind_mat = (label_mat == input_idx)
            ret_mat[ind_mat] = output_idx
        return ret_mat
    # Pad valid class indexing
    ret_valid_class = [False for i in range(output_class_num)]
    for valid_num in valid_class_list:
        ret_valid_class[valid_num] = True
    return label_map_func, ret_valid_class

def get_coco_label_unifier(label_dict):
    return get_label_unifier(label_dict, coco_label_map, final_classes)

def get_fine_grained_ade_label_unifier(label_dict):
    return get_label_unifier(label_dict, ade20k_label_map, final_classes)

def get_hospital_label_unifier(label_dict):
    return get_label_unifier(label_dict, hospital_label_map, final_classes)