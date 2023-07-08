"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-21 13:22:43
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-21 15:36:31
 * @Description: Description
 */
"""

import os
import cv2
import glob
import numpy as np
import math
from mmrotate.core.bbox.transforms import poly2obb_np

def cal_dist(point1, point2):
    return int(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)) + 1

#逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
      angle = (angle/180)*math.pi
      valuex = np.array(valuex)
      valuey = np.array(valuey)
      nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
      nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
      return (nRotatex, nRotatey)
#顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
      angle = (angle/180)*math.pi
      valuex = np.array(valuex)
      valuey = np.array(valuey)
      sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
      sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
      return (sRotatex,sRotatey)
#将四个点做映射
def rotatecordiate(angle,rectboxs,pointx,pointy):
      output = []
      for rectbox in rectboxs:
        if angle>0:
          output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
        else:
          output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
      return output

def imagecrop(image, box):
      xs = [x[1] for x in box]
      ys = [x[0] for x in box]
    #   print(xs)
    #   print(min(xs),max(xs),min(ys),max(ys))
      cropimage = image[min(xs):max(xs), min(ys):max(ys)]
    #   print(cropimage.shape)
    #   cv2.imwrite('cropimage.png', cropimage)
      return cropimage

def crop_obj(ann_file, img_file, out_dir):
    rotateimg = cv2.imread(img_file)
    base_name = os.path.splitext(os.path.basename(ann_file))[0]
    with open(ann_file, 'r') as f:
        print("processing {}...".format(ann_file))
        lines = f.readlines()
        for i, line in enumerate(lines):
            x1, y1, x2, y2, x3, y3, x4, y4, class_name, _ = line.strip().split()
            polys = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32)
            # rect = poly2obb_np(polys, version="oc") # [x_ctr,y_ctr,w,h,angle]
            bboxps = np.array(polys).reshape((4, 2))
            rect = cv2.minAreaRect(bboxps)
            box_origin = cv2.boxPoints(rect)

            M = cv2.getRotationMatrix2D(rect[0],rect[2], 1)
            dst = cv2.warpAffine(rotateimg, M, (2*rotateimg.shape[0], 2*rotateimg.shape[1]))
            
            box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
            cropped_img = imagecrop(dst,np.int0(box))

            class_dir = os.path.join(out_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            if np.all(cropped_img.shape):
                cv2.imwrite(os.path.join(class_dir, "{}_{}.jpeg".format(base_name, i)), cropped_img)
            else:
                continue
            

def main(ann_dir, img_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ann_files = glob.glob(os.path.join(ann_dir, "*.txt"))
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    ann_files = sorted(ann_files)
    img_files = sorted(img_files)

    for ann_file, img_file in zip(ann_files, img_files):
        ann_name = os.path.splitext(os.path.basename(ann_file))[0]
        img_name = os.path.splitext(os.path.basename(img_file))[0]

        assert ann_name == img_name

        crop_obj(ann_file, img_file, out_dir)



if __name__ == "__main__":

    ann_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/annotations/train00001.txt"
    img_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/images/train00001.jpg"

    ann_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/annotations/"
    img_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/images/"
    out_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/eval/crop"
    main(ann_dir, img_dir, out_dir)

    ann_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train/annotations/"
    img_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train/images/"
    out_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train/crop"
    main(ann_dir, img_dir, out_dir)


