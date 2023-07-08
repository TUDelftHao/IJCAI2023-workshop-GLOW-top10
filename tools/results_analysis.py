"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-06-04 21:34:18
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-25 23:37:48
 * @Description: 红色多检，蓝色漏检，黄色错检
 */
"""

import argparse
import os
import cv2

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms_rotated
from mmdet.datasets import build_dataset
from PIL import Image, ImageDraw, ImageFont
from mmrotate.core.bbox.transforms import obb2poly_np
from mmcv.visualization.color import color_val
from mmcv import Config, DictAction


from mmrotate.core.bbox import rbbox_overlaps

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        '--save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    
    args = parser.parse_args()
    return args

def analyze_per_img_dets(gt_bboxes,
                         gt_labels,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    true_positives = np.zeros_like(gt_labels)
    gt_bboxes = torch.from_numpy(gt_bboxes).float()
    
    tp_boxes = []
    fp_boxes = []
    fn_boxes = []
    other = []
    for det_label, det_bboxes in enumerate(result):
        det_bboxes = torch.from_numpy(det_bboxes).float()
        if nms_iou_thr:
            det_bboxes, _ = nms_rotated(
                det_bboxes[:, :5],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = rbbox_overlaps(det_bboxes[:, :5], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[5]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                            tp_boxes.append((det_bbox, det_label))
                        else:
                            other.append((det_bbox, det_label))
                if det_match == 0:  # BG FP
                    fp_boxes.append((det_bbox, det_label))

    for num_tp, gt_label, gt_bbox in zip(true_positives, gt_labels, gt_bboxes):
        if num_tp == 0:  # FN
            fn_boxes.append((gt_bbox, gt_label))

    return tp_boxes, fp_boxes, fn_boxes, other

def plot(
        cfg,
        dataset,
        results,
        score_thr=0,
        nms_iou_thr=None,
        tp_iou_thr=0.5,
        save_dir="./det_analysis"
    ):
    # num_classes = len(dataset.CLASSES)
    os.makedirs(save_dir, exist_ok=True)
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        img_info = dataset.data_infos[idx]
        image_name = img_info['filename']
        img_file = os.path.join(cfg.data.test.img_prefix, image_name)
        ori_img = cv2.imread(img_file)
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        tp_boxes, fp_boxes, fn_boxes, other_boxes = analyze_per_img_dets(gt_bboxes, labels, res_bboxes,
                             score_thr, tp_iou_thr, nms_iou_thr)

        fontpath = '/mnt/cfs_bj/nihao/data/SimHei.ttf'
        font = ImageFont.truetype(fontpath, 18)

        for tp_box in tp_boxes:
            box, label = tp_box
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]
            point1, point2, point3, point4 = [int(x1), int(y1)], [int(x2), int(y2)], \
                                            [int(x3), int(y3)], [int(x4), int(y4)]
            class_name = dataset.CLASSES[label]
            pts = np.array([point1, point2, point3, point4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 255, 0)
            cv2.polylines(ori_img, [pts], True, color, thickness=2)
            cv2.circle(ori_img, tuple(point1), 2, (0,0,255), -1)

            img_pil = Image.fromarray(ori_img)
            draw = ImageDraw.Draw(img_pil)

            draw.text((point1[0] - 20, point1[1] - 20), class_name, font=font, fill=color_val('green'))
            ori_img = np.array(img_pil)

        for fp_box in fp_boxes:
            box, label = fp_box
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]
            point1, point2, point3, point4 = [int(x1), int(y1)], [int(x2), int(y2)], \
                                            [int(x3), int(y3)], [int(x4), int(y4)]
            class_name = dataset.CLASSES[label]
            pts = np.array([point1, point2, point3, point4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 0, 255)
            cv2.polylines(ori_img, [pts], True, color, thickness=2)
            cv2.circle(ori_img, tuple(point1), 2, (0,0,255), -1)

            img_pil = Image.fromarray(ori_img)
            draw = ImageDraw.Draw(img_pil)

            draw.text((point1[0] - 20, point1[1] - 20), class_name, font=font, fill=color_val('red'))
            ori_img = np.array(img_pil)

        for fn_box in fn_boxes:
            box, label = fn_box
            box = torch.cat((box, torch.from_numpy(np.array([1]))), dim=0)
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]
            point1, point2, point3, point4 = [int(x1), int(y1)], [int(x2), int(y2)], \
                                            [int(x3), int(y3)], [int(x4), int(y4)]
            class_name = dataset.CLASSES[label]
            pts = np.array([point1, point2, point3, point4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (255, 0, 0)
            cv2.polylines(ori_img, [pts], True, color, thickness=2)
            cv2.circle(ori_img, tuple(point1), 2, (0,0,255), -1)

            img_pil = Image.fromarray(ori_img)
            draw = ImageDraw.Draw(img_pil)

            draw.text((point1[0] - 20, point1[1] - 20), class_name, font=font, fill=color_val('blue'))
            ori_img = np.array(img_pil)
    
        for other_box in other_boxes:
            box, label = other_box
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]
            point1, point2, point3, point4 = [int(x1), int(y1)], [int(x2), int(y2)], \
                                            [int(x3), int(y3)], [int(x4), int(y4)]
            class_name = dataset.CLASSES[label]
            pts = np.array([point1, point2, point3, point4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 255, 255)
            cv2.polylines(ori_img, [pts], True, color, thickness=2)
            cv2.circle(ori_img, tuple(point1), 2, (0,0,255), -1)

            img_pil = Image.fromarray(ori_img)
            draw = ImageDraw.Draw(img_pil)

            draw.text((point1[0] - 20, point1[1] - 20), class_name, font=font, fill=color_val('yellow'))
            ori_img = np.array(img_pil)

        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, ori_img)
        prog_bar.update()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    

    results = mmcv.load(args.prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)

    plot(
        cfg,
        dataset,
        results,
        score_thr=args.score_thr,
        tp_iou_thr=args.tp_iou_thr,
        save_dir=args.save_dir
    )
    

if __name__ == "__main__":
    main()





        






        

        
            
                         