import os
import collections
import cv2
# import xml.etree.ElementTree as ET
import numpy as np
import glob 
from PIL import Image, ImageDraw, ImageFont
from mmcv.visualization.color import color_val


def txt_parse(txt_anno_file):
    with open(txt_anno_file, 'r') as f:
        lines = f.readlines()
    
    annos = []
    if len(lines) > 0:
        for line in lines:
            if len(line.strip().split()) > 4:
                x1, y1, x2, y2, x3, y3, x4, y4, label, _ = line.strip().split()
                annos.append([[int(float(x1)), int(float(y1))], [int(float(x2)), int(float(y2))],\
                    [int(float(x3)), int(float(y3))], [int(float(x4)), int(float(y4))], label])

    return annos

def draw_annos(anno, ori_img):
    point1, point2, point3, point4, label = anno
    pts = np.array([point1, point2, point3, point4], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(ori_img, [pts], True, (0, 255, 255), thickness=2)
    cv2.circle(ori_img, tuple(point1), 2, (0,0,255), -1)

    # fontpath = '/test/baidu/SimHei.ttf'
    # font = ImageFont.truetype(fontpath, 16)
    img_pil = Image.fromarray(ori_img)
    draw = ImageDraw.Draw(img_pil)

    draw.text((point1[0]-10, point1[1]-20), label, fill=color_val('yellow'))
    # draw.text((point1[0]-10, point1[1]-20), label, font=font)
    img = np.array(img_pil)

    return img

def vis_gen(txt_dir, img_dir, output_dir, img_prefix='.jpg'):
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    for i, txt_file in enumerate(txt_files):
        if i > 100:
            return
        name = os.path.splitext(os.path.basename(txt_file))[0]
        img_path = os.path.join(img_dir, name + img_prefix)
        if os.path.isfile(img_path):
            print('drawing {}'.format(name))
            annos = txt_parse(txt_file)
            img = cv2.imread(img_path)
            for anno in annos:
                img = draw_annos(anno, img)

        res_img_path = os.path.join(output_dir, name + img_prefix)
        cv2.imwrite(res_img_path, img)   

    
if __name__ == '__main__':
    # txt_dir = '/test/database/xinyang/train_eval_data/train_eval_data/labelTxt_eval'
    # img_dir = '/test/database/xinyang/train_eval_data/train_eval_data/images'
    # output_dir = '/test/database/xinyang/train_eval_data/train_eval_data/eval_vis/'
    txt_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations_dota'
    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/images'

    txt_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_scale_05/annotations'
    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_scale_05/images'

    txt_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_lsk_s_ema_fpn_1x_dota_le90_tta/dota/'
    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/debug/images/'
    output_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/debug_vis/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vis_gen(txt_dir, img_dir, output_dir)
                


