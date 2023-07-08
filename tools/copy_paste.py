"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-06-04 22:59:36
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-26 00:43:19
 * @Description: Description
 */
"""

import os
import cv2
import numpy as np
import random
import shutil
import mmcv
import torch
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
from mmrotate.core.bbox.transforms import poly2obb
# random.seed(42)

def paste(bg_img, tar_img, scale_percent=1, return_coord=False, dst_path="./"):
    """
    """

    # bg_name = os.path.basename(src_path1)
    # os.makedirs(dst_path, exist_ok=True)

    # bg_img = cv2.imread(src_path1)
    # tar_img = cv2.imread(src_path2)

    bg_height, bg_width, _ = bg_img.shape
    # print("bg size: width: {}, height: {}".format(bg_width, bg_height))

    width = int(tar_img.shape[1] * scale_percent)
    height = int(tar_img.shape[0] * scale_percent)
    # print("target size: width {}, height {}".format(width, height))

    ############## 背景图片旋转 ##############
    # 计算旋转中心点
    center = (bg_width // 2, bg_height // 2)

    # 创建旋转矩阵
    angle = random.randint(-90, 90)
    # print("angle: ", angle)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # print("rotation_matrix: ", rotation_matrix)

    # 计算图像新边界
    new_w = int(bg_height * np.abs(rotation_matrix[0, 1]) + bg_width * np.abs(rotation_matrix[0, 0]))
    new_h = int(bg_height * np.abs(rotation_matrix[0, 0]) + bg_width * np.abs(rotation_matrix[0, 1]))
    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += (new_w - bg_width) / 2
    rotation_matrix[1, 2] += (new_h - bg_height) / 2

    # 应用旋转矩阵
    rotated_region = cv2.warpAffine(bg_img, rotation_matrix, (new_w, new_h))

    # 保存旋转后的背景图
    # dst_file = os.path.join(dst_path, "rotated_region.jpg")
    # cv2.imwrite(dst_file, rotated_region)

    ############## 拷贝目标至旋转后的图片中 ##############

    center_width = int(new_w * 0.25)  
    center_height = int(new_h * 0.25)  

    left = new_w // 2 - center_width  # 左上角 x 坐标
    top = new_h // 2 - center_height  # 左上角 y 坐标
    right = new_w // 2 + center_width  # 右下角 x 坐标
    bottom = new_h // 2 + center_height  # 右下角 y 坐标

    rand_left = random.randint(left, right // 2)
    rand_top = random.randint(top, bottom // 2)

    scale_w = scale_h = 1.
    if right - rand_left < width:
        scale_w = (right - rand_left) / (width * 1.5)
    if bottom - rand_top < height:
        scale_h = (bottom - rand_top) / (height * 1.5)

    resized_w = int(width * scale_w)
    resized_h = int(height * scale_h)
    resized_tar_img = cv2.resize(tar_img, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    resized_src_mask = np.ones(resized_tar_img.shape, resized_tar_img.dtype) * 255
    patching_center = (int(rand_left + resized_w // 2), \
                           int(rand_top + resized_h // 2))
    # 高斯贴合
    # rotated_region = cv2.seamlessClone(resized_tar_img, rotated_region, resized_src_mask, patching_center, cv2.MIXED_CLONE)
    # 直接贴上
    rotated_region[rand_top:rand_top+resized_h, rand_left:rand_left+resized_w] = resized_tar_img

    # dst_file = os.path.join(dst_path, "rotated_paste_region.jpg")
    # cv2.imwrite(dst_file, rotated_region)

    ############## 图片旋转回原图大小 ##############
    # 计算逆透视变换矩阵
    inverse_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)

    # 进行逆透视变换
    restored_region = cv2.warpAffine(rotated_region, inverse_rotation_matrix, (bg_width, bg_height))



    ############## 获得paste图片的坐标 ##############
    transformed_points = None
    if return_coord:
        # 将原图中的点的坐标转换为旋转后图像中的坐标
        original_points = np.array([[rand_left, rand_top], # 左上角
                                    [rand_left+resized_w, rand_top], # 右上角
                                    [rand_left+resized_w, rand_top+resized_h], #右下角
                                    [rand_left, rand_top+resized_h] #左下角
                                    ], dtype=np.float32)

        # 将原图中的点的坐标转换为旋转后图像中的坐标
        transformed_points = cv2.transform(original_points.reshape(-1, 1, 2), inverse_rotation_matrix).astype(np.int32)

        # 可视化
        # cv2.polylines(restored_region, [transformed_points], True, (0, 255, 255), thickness=2)
        # cv2.circle(restored_region, tuple([transformed_points[0, 0, 0], transformed_points[0, 0, 1]]), 2, (0,0,255), -1)

        # 提取旋转后图像中对应点的坐标
        transformed_points = transformed_points.reshape(-1, 2)

    # 保存结果
    # dst_file = os.path.join(dst_path, "result.jpg")
    # cv2.imwrite(dst_file, restored_region)

    return restored_region, transformed_points


def sync_images(
        bg_dir,
        crop_dir,
        save_dir,
    ):

    save_img_dir = os.path.join(save_dir, "images")
    save_ann_dir = os.path.join(save_dir, "annotations")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_ann_dir, exist_ok=True)

    src_img_dir = os.path.join(bg_dir, "images")
    src_ann_dir = os.path.join(bg_dir, "annotations")
    img_names = os.listdir(src_img_dir)

    # fp_candidates = ["knife", "lighter", "battery", "electronicequipment"]
    # fn_candidates = ["knife", "lighter", "battery", "electronicequipment", "OCbottle", "glassbottle"]

    fp_candidates = ["knife", "lighter"]
    fn_candidates = ["knife", "lighter"]
    prog_bar = mmcv.ProgressBar(len(img_names))

    for k, img_name in enumerate(img_names):
        # if k == 2:
        #     break
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(src_img_dir, img_name)
        ann_path = os.path.join(src_ann_dir, base_name + ".txt")
        bg_img = cv2.imread(img_path)

        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        annos = []
        ann_torch = []
        if len(lines) > 0:
            for line in lines:
                if len(line.strip().split()) > 4:
                    x1, y1, x2, y2, x3, y3, x4, y4, label, _ = line.strip().split()
                    annos.append([x1, y1, x2, y2, x3, y3, x4, y4, label, "0"])
                    ann_torch.append([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)])
        
        ann_torch = torch.from_numpy(np.array(ann_torch, dtype=np.float32))
        ann_torch = poly2obb(ann_torch, version="le90")

        # 添加多检fp，不需要标注信息
        i = 0
        fp_count = random.randint(0, 3)
        try_times = 50
        while i < fp_count:
            bg_img_copy = bg_img.copy()
            cls_idx = random.randint(0, len(fp_candidates)-1)
            cls_name = fp_candidates[cls_idx]
            cls_dir = os.path.join(crop_dir, "fp", cls_name)
            cls_img_names = os.listdir(cls_dir)
            img_idx = random.randint(0, len(cls_img_names)-1)
            cls_img_name = cls_img_names[img_idx]
            cls_img_file = os.path.join(cls_dir, cls_img_name)
            tar_img = cv2.imread(cls_img_file)
            bg_img, points = paste(bg_img, tar_img, scale_percent=1, return_coord=True)
            # 计算nms防止overlap
            point1, point2, point3, point4 = points
            torch_point = torch.from_numpy(np.array([point1[0], point1[1], point2[0], point2[1],
                                point3[0], point3[1], point4[0], point4[1]], dtype=np.float32))
            torch_point = poly2obb(torch_point, version="le90")
            cnt = 0
            success = False
            while cnt < try_times:
                overlaps = rbbox_overlaps(ann_torch, torch_point.unsqueeze(0))
                # print("overlaps: ", overlaps)
                if torch.any(overlaps > 0.3):
                    bg_img, points = paste(bg_img, tar_img, scale_percent=1, return_coord=True)
                    # 计算nms防止overlap
                    point1, point2, point3, point4 = points
                    torch_point = torch.from_numpy(np.array([point1[0], point1[1], point2[0], point2[1],
                                        point3[0], point3[1], point4[0], point4[1]], dtype=np.float32))
                    torch_point = poly2obb(torch_point, version="le90")
                    cnt += 1
                else:
                    ann_torch = torch.cat((ann_torch, torch_point))
                    success = True
                    break
            
            if not success:
                bg_img = bg_img_copy
            i += 1
        
        # 添加漏检fn, 需要标注信息
        fn_count = random.randint(0, 3)
        j = 0
        while j < fn_count:
            bg_img_copy = bg_img.copy()
            cls_idx = random.randint(0, len(fn_candidates)-1)
            cls_name = fn_candidates[cls_idx]
            cls_dir = os.path.join(crop_dir, "fn", cls_name)
            cls_img_names = os.listdir(cls_dir)
            img_idx = random.randint(0, len(cls_img_names)-1)
            cls_img_name = cls_img_names[img_idx]
            cls_img_file = os.path.join(cls_dir, cls_img_name)
            tar_img = cv2.imread(cls_img_file)
            bg_img, points = paste(bg_img, tar_img, scale_percent=1, return_coord=True)
            point1, point2, point3, point4 = points

            torch_point = torch.from_numpy(np.array([point1[0], point1[1], point2[0], point2[1],
                                point3[0], point3[1], point4[0], point4[1]], dtype=np.float32))
            torch_point = poly2obb(torch_point, version="le90")
            cnt = 0
            success = False
            while cnt < try_times:
                overlaps = rbbox_overlaps(ann_torch, torch_point.unsqueeze(0))
                # print("overlaps: ", overlaps)
                if torch.any(overlaps > 0.3):
                    bg_img, points = paste(bg_img, tar_img, scale_percent=1, return_coord=True)
                    # 计算nms防止overlap
                    point1, point2, point3, point4 = points
                    torch_point = torch.from_numpy(np.array([point1[0], point1[1], point2[0], point2[1],
                                        point3[0], point3[1], point4[0], point4[1]], dtype=np.float32))
                    torch_point = poly2obb(torch_point, version="le90")
                    cnt += 1
                else:
                    ann_torch = torch.cat((ann_torch, torch_point))
                    success = True
                    annos.append([str(int(point1[0])), str(int(point1[1])), 
                          str(int(point2[0])), str(int(point2[1])),
                          str(int(point3[0])), str(int(point3[1])),
                          str(int(point4[0])), str(int(point4[1])),
                          cls_name,
                          "0"
                          ])
                    break
            if not success:
                bg_img = bg_img_copy
            j += 1

        cv2.imwrite(os.path.join(save_img_dir, img_name), bg_img)
        save_txt_file = os.path.join(save_ann_dir, base_name + ".txt")
        with open(save_txt_file, 'w') as f:
            for anno in annos:
                f.write(anno[0] + " " + anno[1] + " " + anno[2] + " "
                        + anno[3] + " " + anno[4] + " " + anno[5] + " "
                        + anno[6] + " " + anno[7] + " " + anno[8] + " " 
                        + anno[9] + " " + "\n")

        prog_bar.update()
        

if __name__ == "__main__":
    # paste(
    #     "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train_k_folds/fold_0/images/train00003.jpg",
    #     "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/crop_cls/train/knife/train00965_4.jpeg", 
    #     "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train_k_folds/fold_0_paste/images/"
    #     )
    # bg_img = cv2.imread("/Users/nihao/data/ICJAI/test_phase2/images/test02001.jpg")
    # tar_img = cv2.imread("/Users/nihao/data/ICJAI/validation/knife/train01981_0.jpeg")
    # paste(
    #     bg_img,
    #     tar_img, 
    #     dst_path="./images/paste_demo/",
    #     return_coord=True
    #     )

    for i in range(5):
        bg_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_k_folds/fold_{}/".format(i)
        crop_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/submit_20230530/pred_crop/"
        save_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_k_folds/fold_{}_paste_knife-lighter/".format(i)

        sync_images(bg_dir, crop_dir, save_dir)
