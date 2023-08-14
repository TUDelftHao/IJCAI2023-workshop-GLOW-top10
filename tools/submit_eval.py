"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-20 11:25:47
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-20 15:50:59
 * @Description: Description
 */
"""
import glob
import os
import numpy as np
import os.path as osp

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np


CLASSES = (
        'OCbottle', 'battery', 'lighter', 'electronicequipment', 
        'umbrella', 'metalbottle', 'pressure', 'glassbottle', 'knife'
    )

def get_annotations(ann_folder, version="le90", prefix=".jpg"):
    cls_map = {c: i
                   for i, c in enumerate(CLASSES)
            }  # in mmdet v2.0 label is 0-based
    ann_files = glob.glob(ann_folder + '/*.txt')
    data_infos = []

    for ann_file in ann_files:
        data_info = {}
        img_id = osp.split(ann_file)[1][:-4]
        img_name = img_id + prefix
        data_info['filename'] = img_name
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []

        if os.path.getsize(ann_file) == 0:
            continue

        with open(ann_file) as f:
            s = f.readlines()
            for si in s:
                bbox_info = si.split()
                poly = np.array(bbox_info[:8], dtype=np.float32)
                try:
                    x, y, w, h, a = poly2obb_np(poly, version)
                except:  # noqa: E722
                    continue
                cls_name = bbox_info[8]
                difficulty = int(bbox_info[9])
                label = cls_map[cls_name]
                
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)

        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                    dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)
        
        data_info['ann']['bboxes_ignore'] = np.zeros(
            (0, 5), dtype=np.float32)
        data_info['ann']['labels_ignore'] = np.array(
            [], dtype=np.int64)
        data_info['ann']['polygons_ignore'] = np.zeros(
            (0, 8), dtype=np.float32)

        data_infos.append(data_info)
    img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]

    return data_infos, img_ids

def get_dets(det_file, version="le90"):
    cls_map = {c: i for i, c in enumerate(CLASSES)}
    
    img_set = dict()
    with open(det_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name, class_name, score, x1, y1, x2, y2, x3, y3, x4, y4 = line.strip().split()
            class_id = cls_map[class_name]
            poly = np.array(list(map(float, [x1, y1, x2, y2, x3, y3, x4, y4])), dtype=np.float32)
            x, y, w, h, a = poly2obb_np(poly, version)
            rbbox = np.array([x, y, w, h, a, score], dtype=np.float32)
            rbbox = rbbox[np.newaxis, :]

            if file_name not in img_set:
                img_set[file_name] = [np.zeros((0, 6), dtype=np.float32) for _ in range(len(cls_map))]
                img_set[file_name][class_id] = rbbox
            else:
                if img_set[file_name][class_id].shape[0] == 0:
                    img_set[file_name][class_id] = rbbox
                else:
                    img_set[file_name][class_id] = np.concatenate((img_set[file_name][class_id], rbbox))
                        
    dets = []
    for img_file in img_set:
        dets.append(img_set[img_file])

    return dets, img_set


if __name__ == "__main__":
    gt_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/annotations/"
    # det_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90/submit_results/results.txt"
    det_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/merge_submit_results/merged_results.txt"

    det_results, img_set = get_dets(det_file)
    data_infos, img_ids = get_annotations(gt_dir)
    annotations = [data_infos[i]['ann'] for i in range(len(det_results))]
    mean_ap, eval_results = eval_rbbox_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=0.5,
        use_07_metric=True,
        dataset=CLASSES,
        logger=None,
        nproc=4
    )