"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-21 16:49:55
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-24 13:42:59
 * @Description: Description
 */
"""

import sys
import torch
import os
from collections import OrderedDict
from pathlib import Path
import argparse
from collections import defaultdict
from torch import Tensor, nn
import copy

def merge(source_ckpt, tobe_merge_ckpt, new_ckpt_file, is_same_backbone=True):
    source_model = torch.load(source_ckpt, map_location=torch.device('cpu'))
    tobe_merge = torch.load(tobe_merge_ckpt, map_location=torch.device('cpu'))

    new_model = copy.deepcopy(source_model)

    if is_same_backbone:
        for key in new_model['state_dict']:
            if key.startswith("backbone"):
                tb_key_list = key.split(".")
                tb_key = '.'.join(tb_key_list[1:])
                if tb_key in tobe_merge["state_dict"]:
                    new_model['state_dict'][key] = tobe_merge["state_dict"][tb_key]
    else:
        for key in list(new_model['state_dict'].keys()):
            if key.startswith("backbone"):
                del new_model['state_dict'][key]
        
        for key in tobe_merge["state_dict"]:
            if not key.startswith("head"):
                new_model['state_dict']["backbone." + key] = tobe_merge["state_dict"][key]

    torch.save(new_model, new_ckpt_file) 
    print("done")

if __name__ == "__main__":
    # source_ckpt = "/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth"
    # tobe_merge_ckpt =  "/mnt/cfs_bj/nihao/data/ICJAI2023/output/20230521-161033-lsk_s-224/checkpoint-84.pth.tar"
    # new_ckpt_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/self_pretrain_merge_dota_lsknet_s_224.pth"

    source_ckpt = "/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth"
    tobe_merge_ckpt =  "/mnt/cfs_bj/nihao/data/ICJAI2023/output/backbone_pretrain/20230524-095324-lsk_m-224/model_best.pth.tar"
    new_ckpt_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/self_pretrain_merge_dota_lsknet_m_224.pth"
    merge(source_ckpt, tobe_merge_ckpt, new_ckpt_file, is_same_backbone=False)

