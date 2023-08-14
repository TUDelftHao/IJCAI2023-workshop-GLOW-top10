"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-20 17:07:37
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-21 19:56:07
 * @Description: Description
 */
"""
import os
import glob
import shutil

def generate_k_folder(ori_dir, out_dir, k=5):
    if not os.path.join(out_dir):
        os.makedirs(out_dir)
    
    ori_ann_dir = os.path.join(ori_dir, "annotations_dota")
    ori_img_dir = os.path.join(ori_dir, "images")

    ori_anns = glob.glob(os.path.join(ori_ann_dir, "*.txt"))
    ori_imgs = glob.glob(os.path.join(ori_img_dir, "*.jpg"))

    ori_anns = sorted(ori_anns)
    ori_imgs = sorted(ori_imgs)

    k_samples = len(ori_anns) // k

    for i in range(k):
        if i != k-1:
            ori_anns_k = ori_anns[i * k_samples: (i+1) * k_samples]
            ori_imgs_k = ori_imgs[i * k_samples: (i+1) * k_samples]
        else:
            ori_anns_k = ori_anns[i * k_samples:]
            ori_imgs_k = ori_imgs[i * k_samples:]

        target_dir = os.path.join(out_dir, "fold_{}".format(i))
        target_img_dir = os.path.join(target_dir, "images")
        target_ann_dir = os.path.join(target_dir, "annotations")
        if not os.path.exists(target_img_dir):
            os.makedirs(target_img_dir)
        if not os.path.exists(target_ann_dir):
            os.makedirs(target_ann_dir)

        ori_anns_k = sorted(ori_anns_k)
        ori_imgs_k = sorted(ori_imgs_k)

        for ori_ann, ori_img in zip(ori_anns_k, ori_imgs_k):
            ann_file_name = os.path.splitext(os.path.basename(ori_ann))[0]
            img_file_name = os.path.splitext(os.path.basename(ori_img))[0]

            assert ann_file_name == img_file_name

            tar_ann_file = os.path.join(target_ann_dir, ann_file_name + ".txt")
            tar_img_file = os.path.join(target_img_dir, img_file_name + ".jpg")

            shutil.copyfile(ori_ann, tar_ann_file)
            shutil.copyfile(ori_img, tar_img_file)

if __name__ == "__main__":
    ori_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/"
    out_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_k_folds"

    generate_k_folder(ori_dir, out_dir)



            
