"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-13 23:33:17
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-01 23:24:59
 * @Description: Description
 */
"""

import os
import glob
import shutil
import numpy as np

def convert(ori_dir, tar_dir):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    ori_annos = glob.glob(os.path.join(ori_dir, "*.txt"))
    classes = set()

    for ori_anno in ori_annos:

        ori_anno_name = os.path.basename(ori_anno)

        new_anno = []
        with open(ori_anno, 'r') as f:
            lines = f.readlines()

        for line in lines:
            file_name, class_name, x1, y1, x2, y2, x3, y3, x4, y4 = line.strip().split()
            if class_name not in classes:
                classes.add(class_name)
            
            new_anno.append(map(str, [x1, y1, x2, y2, x3, y3, x4, y4, class_name, 0]))

        # 这里不能用file_name，原始标注文件里的名字可能有问题，结果会少一个
        new_file = os.path.join(tar_dir, ori_anno_name)
        with open(new_file, 'w') as f:
            for ann in new_anno:
                ann_str = ' '.join(ann)
                # print("new ann: ", ann_str)
                f.write(ann_str + '\n')

    return classes

def path_gen(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def split_trainval(base_dir):
    train_img_dir = os.path.join(base_dir, "train/images")
    train_ann_dir = os.path.join(base_dir, "train/annotations")
    eval_img_dir = os.path.join(base_dir, "eval/images")
    eval_ann_dir = os.path.join(base_dir, "eval/annotations")
    path_gen(train_img_dir)
    path_gen(train_ann_dir)
    path_gen(eval_img_dir)
    path_gen(eval_ann_dir)

    
    all_anno_dir = os.path.join(base_dir, "annotations_dota")
    all_image_dir = os.path.join(base_dir, "images")

    all_annos = glob.glob(os.path.join(all_anno_dir, "*.txt"))
    all_images = glob.glob(os.path.join(all_image_dir, "*.jpg"))

    for ann in all_annos:
        file_name = os.path.splitext(os.path.basename(ann))[0]
        image_name = os.path.join(all_image_dir, file_name + ".jpg")
        
        if image_name in all_images:
            val = np.random.uniform()
            if val <= 0.7:
                tar_anno = os.path.join(train_ann_dir, file_name + ".txt")
                tar_img = os.path.join(train_img_dir, file_name + ".jpg")
            else:
                tar_anno = os.path.join(eval_ann_dir, file_name + ".txt")
                tar_img = os.path.join(eval_img_dir, file_name + ".jpg")

            shutil.copyfile(ann, tar_anno)
            shutil.copyfile(image_name, tar_img)

if __name__ == "__main__":
    # ori_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations"
    # tar_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations_dota"

    ori_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_lsk_s_ema_fpn_1x_dota_le90_tta/icjai/"
    tar_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_lsk_s_ema_fpn_1x_dota_le90_tta/dota/"

    classes = convert(ori_dir, tar_dir)
    print("all class are: ", classes)
    print("class num: ", len(classes))

    # split_trainval("/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/")