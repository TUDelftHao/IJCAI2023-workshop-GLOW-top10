"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-06-01 23:04:33
 * @Last Modified by: nihao
 * @Last Modified time: 2023-06-01 23:27:40
 * @Description: Description
 */
"""
import os

def convert(source_file, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    with open(source_file, 'r') as f:
        lines = f.readlines()

    txt_names = {}
    res = ""
    for i, line in enumerate(lines):
        txt_name, class_name, score, x1, y1, \
            x2, y2, x3, y3, x4, y4 = line.split()
        base_name = txt_name.split(".")[0]

        if base_name in txt_names:
            txt_names[base_name].append(txt_name + " " + class_name + " " + str(int(float(x1))) + " " + str(int(float(y1))) + " " + str(int(float(x2))) + " " + str(int(float(y2))) + " " \
                + str(int(float(x3))) + " " + str(int(float(y3))) + " " + str(int(float(x4))) + " " + str(int(float(y4))) \
                    + "\n")
        else:
            txt_names[base_name] = [txt_name + " " + class_name + " " + str(int(float(x1))) + " " + str(int(float(y1))) + " " + str(int(float(x2))) + " " + str(int(float(y2))) + " " \
                + str(int(float(x3))) + " " + str(int(float(y3))) + " " + str(int(float(x4))) + " " + str(int(float(y4))) \
                    + "\n"]
    
    for txt in txt_names:
        file_path = os.path.join(target_dir, txt + '.txt')

        with open(file_path, 'w') as f:
            for lin in txt_names[txt]:
                f.writelines(lin)

if __name__ == "__main__":
    source_file = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_lsk_s_ema_fpn_1x_dota_le90_tta/submit_results/results.txt"
    target_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_lsk_s_ema_fpn_1x_dota_le90_tta/icjai/"
        
    convert(source_file, target_dir)

        
        