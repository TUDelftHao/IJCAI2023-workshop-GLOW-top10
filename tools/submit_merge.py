"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-20 13:29:32
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-02 01:38:37
 * @Description: Description
 */
"""
import os
import os.path as osp
import numpy as np
import mmcv

from collections import defaultdict
from submit_eval import get_dets, CLASSES
from mmrotate.datasets.dota import _merge_func
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

from functools import partial

def ger_merge_result(det_files, submission_dir, nproc=4):
    collector = defaultdict(list)
    for det_file in det_files:
        det_results, img_set = get_dets(det_file)
        
        for img_file in img_set:
            new_result = []
            for i, dets in enumerate(img_set[img_file]):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))
            new_result = np.concatenate(new_result, axis=0)
            collector[img_file].append(new_result)

    merge_func = partial(_merge_func, CLASSES=CLASSES, iou_thr=0.3)
    if nproc <= 1:
        print('Single processing')
        merged_results = mmcv.track_iter_progress(
            (map(merge_func, collector.items()), len(collector)))
    else:
        print('Multiple processing')
        merged_results = mmcv.track_parallel_progress(
            merge_func, list(collector.items()), nproc)

    id_list, dets_list = zip(*merged_results)

    result_files = results2submission(id_list, dets_list, submission_dir)

    return result_files

def results2submission(id_list, dets_list, out_folder=None, version='le90'):
    """Generate the submission of full images.

    Args:
        id_list (list): Id of images.
        dets_list (list): Detection results of per class.
        out_folder (str, optional): Folder of submission.
    """

    os.makedirs(out_folder, exist_ok=True)
    res_file = osp.join(out_folder, 'merged_results.txt')
    
    all_res = []
    for img_id, dets_per_cls in zip(id_list, dets_list):
        for i, dets in enumerate(dets_per_cls):
            if dets.size == 0:
                continue
            bboxes = obb2poly_np(dets, version)
            for bbox in bboxes:
                txt_element = [img_id, CLASSES[i], str(bbox[-1])
                            ] + [f'{p:.2f}' for p in bbox[:-1]]
                all_res.append(txt_element)
    
    all_res = sorted(all_res)
    # import pdb; pdb.set_trace()
    with open(res_file, 'w') as f:
        for txt_element in all_res:
            f.writelines(' '.join(txt_element) + '\n')

    return res_file


if __name__ == "__main__":
    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_caraf/submit_results/results.txt"

    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_data_aug_6x_all/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/20230516_084557.txt"


    # 5折训练和全量训练各自的预测结果
    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_fold_0/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_fold_1/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_fold_2/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_fold_3/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_fold_4/results.txt"
    # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_lr2x/submit_results/results.txt"

    # 5折模型平均后的预测结果
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_merge/results.txt"

    # 5折模型+全量训练模型平均后的预测结果
    # det_file_8 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/lsk_s_ema_fpn_1x_dota_le90_k_fold/submit_results_merge_baseline/results.txt"



    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/fold_0/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/fold_1/submit_results/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/fold_2/submit_results/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/fold_3/submit_results/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/fold_4/submit_results/results.txt"
    # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/submit_20230521/submit_results/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230521/merged_model/submit_results/results.txt"

    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_0/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_1/submit_results/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_2/submit_results/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_3/submit_results/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_4/submit_results/results.txt"
    # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_all/submit_results/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_merge/submit_results/results.txt"

    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_0/submit_3/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_1/submit_3/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_2/submit_3/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_3/submit_3/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_4/submit_3/results.txt"
    # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_all/submit_3/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230530/fold_merge/submit_3/results.txt"

    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_0/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_1/submit_results/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_2/submit_results/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_3/submit_results/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_4/submit_results/results.txt"
    # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_all/submit_results/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_merge/submit_results/results.txt"

    # det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_0/submit_results/results.txt"
    # det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_1/submit_results/results.txt"
    # det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_2/submit_results/results.txt"
    # det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_3/submit_results/results.txt"
    # det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_4/submit_results/results.txt"
    # # det_file_6 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604/fold_all/submit_results/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230605/fold_merge/submit_results/results.txt"

    det_file_1 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_0/submit_results/results.txt"
    det_file_2 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_1/submit_results/results.txt"
    det_file_3 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_2/submit_results/results.txt"
    det_file_4 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_3/submit_results/results.txt"
    det_file_5 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_4/submit_results/results.txt"
    # det_file_7 = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_submit_20230604_tta/fold_merge/submit_results/results.txt"


    submission_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/merge_submit_results_20230625_tta/"

    ger_merge_result([det_file_1, det_file_2, det_file_3, det_file_4, det_file_5], submission_dir=submission_dir)