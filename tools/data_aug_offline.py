"""
Used for DOTA style data to augment by rotation
"""

import cv2
import os
import glob
from multiprocessing import Pool
import numpy as np
import math
from imgaug import augmenters as iaa
import shutil

class RotationAug:
    def __init__(
        self,
        in_root,
        out_root,
        aug_times=3,
        num_process=16
    ):
        assert aug_times >= 1

        self.in_images_dir = os.path.join(in_root, 'eval/images/')
        self.in_labels_dir = os.path.join(in_root, 'eval/annotations/')
        self.out_images_dir = os.path.join(out_root, 'eval_aug/images/')
        self.out_labels_dir = os.path.join(out_root, 'eval_aug/annotations/')

        self.angle_list = np.arange(-60, 60, 3)
        print('augmnetaiton lists: ', self.angle_list)
        images = glob.glob(self.in_images_dir + '*.jpg')
        labels = glob.glob(self.in_labels_dir + '*.txt')
        image_ids = [*map(lambda x: os.path.splitext(os.path.split(x)[-1])[0], images)]
        label_ids = [*map(lambda x: os.path.splitext(os.path.split(x)[-1])[0], labels)]
        self.image_ids = image_ids


        if os.path.exists(self.out_images_dir):
            shutil.rmtree(self.out_images_dir)
        if os.path.exists(self.out_labels_dir):
            shutil.rmtree(self.out_labels_dir)

        if not os.path.isdir(out_root):
            os.makedirs(out_root)
        if not os.path.isdir(self.out_images_dir):
            os.makedirs(self.out_images_dir)
        if not os.path.isdir(self.out_labels_dir):
            os.makedirs(self.out_labels_dir)

        self.num_process = num_process
        self.aug_times = aug_times

    def getImgSize(self, img_path):
        img = cv2.imread(img_path)
        h, w, c = img.shape
        center = (h / 2, w / 2)

        return h, w, center
        
    def rotateImg(self, img_path, angle):
        img = cv2.imread(img_path)

        seq = iaa.Sequential([
            iaa.Affine(
                rotate=angle,
                fit_output=True
            )
        ])
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([img])[0]

        return image_aug

    def rotate_bound(self, img_path, angle):
        image = cv2.imread(img_path)

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        aug_img = cv2.warpAffine(image, M, (nW, nH))

        return aug_img, M

    def parseAnnoFile(self, annoTxt):
        with open(annoTxt, 'r') as fin:
            lines = fin.readlines()

        anno_dict = {}
        poly_np = np.zeros((len(lines), 8))
        label_list = []
        diff_list = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2, x3, y3, x4, y4, label, difficult = line.strip().split()
            poly = x1, y1, x2, y2, x3, y3, x4, y4
            poly = [*map(lambda x: float(x), poly)]
            poly_np[i] = x1, y1, x2, y2, x3, y3, x4, y4
            label_list.append(label)
            diff_list.append(difficult)

        anno_dict['polys'] = poly_np
        anno_dict['labels'] = label_list
        anno_dict['difficulties'] = diff_list

        return anno_dict

    def poly2angle(self, poly):
        bboxps = np.array(poly).reshape((4, 2)).astype(np.float32)
        rbbox = cv2.minAreaRect(bboxps)
        x, y, w, h, angle = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]

        return x, y, w, h, angle

    def rotateAnnoPolys(self, img_height, poly, img_center, angle, M):
        rotated_poly = np.zeros_like(poly)
        p1, p2, p3, p4 = poly[..., :2], poly[..., 2:4], poly[..., 4:6], poly[..., 6:8]
        angle = math.radians(angle)

        rotated_poly[..., 0:2] = self.rotate(p1, M)
        rotated_poly[..., 2:4] = self.rotate(p2, M)
        rotated_poly[..., 4:6] = self.rotate(p3, M)
        rotated_poly[..., 6:8] = self.rotate(p4, M)

        return rotated_poly

    def rotate(self, ps, m):
        pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
        pts = np.hstack([pts, np.ones([len(pts), 1])]).T
        target_point = np.dot(m, pts)
        target_point = np.array([[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))])

        return target_point


    def rotateDataSingle(self, name):
        img_path = os.path.join(self.in_images_dir, name + '.jpg')
        anno_path = os.path.join(self.in_labels_dir, name + '.txt')
        if os.path.exists(img_path) and os.path.exists(anno_path):
            name = os.path.splitext(os.path.basename(img_path))[0]
            angles = np.random.choice(self.angle_list, self.aug_times, replace=True)
            anno_dict = self.parseAnnoFile(anno_path)
            polys = anno_dict['polys']
            h, w, center = self.getImgSize(img_path)

            for angle in angles:
                aug_img, M = self.rotate_bound(img_path, angle=angle)
                rotated_poly = self.rotateAnnoPolys(h, polys, center, angle, M)
                # if rotated_poly.shape != [0]:
                #     rotated_poly[:, 0::2] = np.clip(rotated_poly[:, 0::2], 0, h - 1)
                #     rotated_poly[:, 1::2] = np.clip(rotated_poly[:, 1::2], 0, w - 1)
                rotated_poly = rotated_poly.astype(int)
                # import pdb; pdb.set_trace()
                anno_dict['polys'] = rotated_poly
                new_name = name + '_' + str(angle)
                self.rotatedOutput(aug_img, anno_dict, new_name)
            
    def rotatedOutput(self, img, anno_dict, name):
        out_img_path = os.path.join(self.out_images_dir, name + '.jpg')
        out_anno_path = os.path.join(self.out_labels_dir, name + '.txt')
        cv2.imwrite(out_img_path, img)
        poly_np = anno_dict['polys']
        label_list = anno_dict['labels']
        diff_list = anno_dict['difficulties']
        num = len(label_list)

        with open(out_anno_path, 'w') as fwrite:
            print('............', out_anno_path)
            for i in range(num):
                x1, y1, x2, y2, x3, y3, x4, y4 = poly_np[i]
                label = label_list[i]
                difficulty = diff_list[i]
                line = ' '.join(list(map(str, [x1, y1, x2, y2, x3, y3, x4, y4, label, difficulty])))
                line = line.strip()
                print('line: ', line)
                fwrite.writelines(line + '\n')

    def muli_rotate(self):
        # 多线程处理
        with Pool(self.num_process) as p:
            p.map(self.rotateDataSingle, self.image_ids)
        # debug用
        # for img_id in self.image_ids:
        #     self.rotateDataSingle(img_id)

if __name__ == '__main__':
    # valsplit = RotationAug('/test/data/nihao/data/shuke_stack/',
    #                         '/test/data/nihao/data/shuke_stack/aug_rot',
    #                         aug_times=10,
    #                         num_process=32)
    valsplit = RotationAug('/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/',
                            '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/',
                            aug_times=5,
                            num_process=32)
    valsplit.muli_rotate()