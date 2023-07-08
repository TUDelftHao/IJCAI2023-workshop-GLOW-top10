'''
adapted from https://github.com/open-mmlab/mmcv/blob/eaf25af6c4ca18d7f3c74c3849989e517168d9a8/mmcv/image/geometric.py
'''

import numbers
import cv2
import numpy as np
import codecs
import os
import glob

import xml.etree.ElementTree as ET
from xml.dom import minidom 
from xml.etree.ElementTree import Element, SubElement, tostring

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


# ------------------------- xml file processing -------------------------------- #

def getDOTAanno(anno_path):
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    
    annos = []
    if len(lines) > 0:
        for line in lines:
            if len(line.strip().split()) > 4:
                x1, y1, x2, y2, x3, y3, x4, y4, label, _ = line.strip().split()
                annos.append([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)),\
                    int(float(x3)), int(float(y3)), int(float(x4)), int(float(y4)), label])

    return annos

def write_txt(bboxes, dest_path):
    with open(dest_path, 'w') as f:
        for ann in bboxes:
            ann_str = ' '.join(str(it) for it in ann)
            ann_str += ' 0'
            # print("new ann: ", ann_str)
            f.write(ann_str + '\n')

# ------------------------------- image rescale --------------------------------------- #
def _scale_size(scale, *args):

    """Rescale a size by a ratio.
    Args:
        args: size.
        scale (float): Scaling factor.
    Returns:
        list[int]: scaled size.
    """

    # w, h = size
    results = []
    for val in args:
        results.append(int(val * float(scale)))
    # return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)
    return results

def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):

    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """

    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def rescale_size(old_size, scale, return_scale=False):

    """Calculate the new size to be rescaled to.
    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.
    Returns:
        tuple[int]: The new rescaled image size.
    """

    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size(scale_factor, w, h)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

def imrescale(img,
              scale,
              anno_path=None,
              dest_path=None,
              return_boxes=False,
              interpolation='bilinear'):

    """Resize image while keeping the aspect ratio.
    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        anno_path: annotations of images, xml file. default: None
        dest_path: save the rescaled annotations, xml file. if None, and anno_path is not None, the result will be overwrote
        to anno_path. default: None
        interpolation (str): same to imresize.
        
    Returns:
        ndarray: The rescaled image.
    """
    
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, tuple(new_size), interpolation=interpolation)

    if anno_path:
        if not os.path.basename(anno_path).endswith('.txt'):
            raise TypeError('{} is not an XML file'.format(anno_path))

        bboxes = getDOTAanno(anno_path)
        # print("bboxes: ", bboxes)
        
        for bbox in bboxes:
            ori_size = bbox[:8]
            scaled_size = _scale_size(scale_factor, ori_size[0], ori_size[1], ori_size[2], ori_size[3],
                                                    ori_size[4], ori_size[5], ori_size[6], ori_size[7],)
            bbox[:8] = scaled_size

        #rewrite into xml
        if return_boxes:
            return rescaled_img, bboxes
        if dest_path:
            if not os.path.basename(dest_path).endswith('.txt'):
                raise TypeError('{} is not an txt file'.format(dest_path))
            write_txt(bboxes, dest_path)

    return rescaled_img


def begin_rescale(img_dir, txt_dir, output_dir, scaling=None):

    os.makedirs(output_dir, exist_ok=True)

    out_ann_dir = os.path.join(output_dir, "annotations")
    if not os.path.exists(out_ann_dir):
        os.makedirs(out_ann_dir)
    out_img_dir = os.path.join(output_dir, "images")
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    def cal_scale(h, w, long=1333):
        max_long = max(h, w)
        scale = long / max_long * 1.0
        return scale

    ims = glob.glob(os.path.join(img_dir, '*.jpg'))
    # xmls = glob.glob(os.path.join(xml_dir, '*.xml'))
    txts = glob.glob(os.path.join(txt_dir, '*.txt'))
    #assert len(ims) == len(xmls)
    for im in ims:
        # im_name = im.split('/')[-1]
        im_name = os.path.basename(im)
        if im_name == "train00025.jpg":
            continue
        xml_name = os.path.splitext(im_name)[0] + '.txt'
        if os.path.exists(os.path.join(txt_dir, xml_name)):
            img = cv2.imread(im)
            h, w = img.shape[:2]
            if scaling is None:
                scale_factor = cal_scale(h, w)
            else:
                scale_factor = scaling
            anno_path = os.path.join(txt_dir, xml_name)
            xml_out_path = os.path.join(out_ann_dir, xml_name)
            img_resize =  os.path.join(out_img_dir, im_name)
            rescaled_img = imrescale(img, scale=scale_factor, anno_path=anno_path, dest_path=xml_out_path)
            cv2.imwrite(img_resize, rescaled_img)


def begin_per_img(img_path, xml_path):
    def cal_scale(h, w, long=1333):
        max_long = max(h, w)
        scale = long / max_long * 1.0
        return scale
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale_factor = cal_scale(h, w)
    rescaled_img, rescaled_box = imrescale(img, scale=scale_factor, anno_path=xml_path, return_boxes=True)
    return rescaled_img, rescaled_box


if __name__ == '__main__':
    
    # image_path = '00177.jpg'
    # anno_path = '00177.xml'
    # anno_path = None
    # dest_path = 'result.xml'
    # dest_path = None
    # scale_factor = 0.7

    # img_dir = '/test/database/gongniukaiguan/paddle-job-985131-0/train_data/images/'
    # xml_dir = '/test/database/gongniukaiguan/paddle-job-985131-0/train_data/anno'
    # output_dir = '/test/database/gongniukaiguan/paddle-job-985131-0_small/'

    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/images/'
    xml_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations_dota/'
    output_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_scale_05'

    begin_rescale(img_dir, xml_dir, output_dir, scaling=0.5)


    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/images/'
    xml_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations_dota/'
    output_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_scale_15'
    begin_rescale(img_dir, xml_dir, output_dir, scaling=1.5)
