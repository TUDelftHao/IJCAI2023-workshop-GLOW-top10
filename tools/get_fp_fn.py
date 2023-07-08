"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-06-25 22:51:49
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-25 23:40:17
 * @Description: Description
 */
"""
import os
import mmcv
import torch 
import cv2 
import numpy as np

from results_analysis import analyze_per_img_dets, parse_args
from obj_crop import rotatecordiate, imagecrop
from mmrotate.core.bbox.transforms import obb2poly_np
from mmdet.datasets import build_dataset
from mmcv import Config, DictAction


def crop_fp_fn(
    cfg,
    dataset,
    results,
    score_thr=0,
    nms_iou_thr=None,
    tp_iou_thr=0.5,
    save_dir="./det_analysis"
):
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
        base_name = os.path.splitext(image_name)[0]

        ori_img = cv2.imread(img_file)
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        tp_boxes, fp_boxes, fn_boxes, other_boxes = analyze_per_img_dets(gt_bboxes, labels, res_bboxes,
                             score_thr, tp_iou_thr, nms_iou_thr)


        # 多检
        for i, fp_box in enumerate(fp_boxes):
            box, label = fp_box
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]

            class_name = dataset.CLASSES[label]
            polys = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32)
            bboxps = np.array(polys).reshape((4, 2))
            rect = cv2.minAreaRect(bboxps)
            box_origin = cv2.boxPoints(rect)

            M = cv2.getRotationMatrix2D(rect[0],rect[2], 1)
            dst = cv2.warpAffine(ori_img, M, (2 * ori_img.shape[0], 2 * ori_img.shape[1]))
            
            box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
            cropped_img = imagecrop(dst,np.int0(box))

            class_dir = os.path.join(save_dir, "fp", class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            if np.all(cropped_img.shape):
                cv2.imwrite(os.path.join(class_dir, "{}_{}.jpeg".format(base_name, i)), cropped_img)
            else:
                continue

        # 漏检
        for i, fn_box in enumerate(fn_boxes):
            box, label = fn_box
            box = torch.cat((box, torch.from_numpy(np.array([1]))), dim=0)
            poly = obb2poly_np(np.expand_dims(box.numpy()[:6], axis=0), version='le90')
            x1, y1, x2, y2, x3, y3, x4, y4, score = poly[0]

            class_name = dataset.CLASSES[label]
            polys = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32)
            bboxps = np.array(polys).reshape((4, 2))
            rect = cv2.minAreaRect(bboxps)
            box_origin = cv2.boxPoints(rect)

            M = cv2.getRotationMatrix2D(rect[0],rect[2], 1)
            dst = cv2.warpAffine(ori_img, M, (2 * ori_img.shape[0], 2 * ori_img.shape[1]))
            
            box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
            cropped_img = imagecrop(dst,np.int0(box))

            class_dir = os.path.join(save_dir, "fn", class_name)

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            if np.all(cropped_img.shape):
                cv2.imwrite(os.path.join(class_dir, "{}_{}.jpeg".format(base_name, i)), cropped_img)
            else:
                continue

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

    crop_fp_fn(
        cfg,
        dataset,
        results,
        score_thr=args.score_thr,
        tp_iou_thr=args.tp_iou_thr,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()