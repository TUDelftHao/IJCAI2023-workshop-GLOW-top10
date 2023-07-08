"""
/*
 * @Author: nihao
 * @Email: nihao@baidu.com
 * @Date: 2023-05-20 16:01:36
 * @Last Modified by: nihao
 * @Last Modified time: 2023-05-21 11:49:20
 * @Description: Description
 */
"""

import os
from argparse import ArgumentParser

import torch

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--model_dir', nargs='+', help='the directory where checkpoints are saved')
    # parser.add_argument(
    #     '--model_names', nargs='+'
    # )
    parser.add_argument(
        '--save_dir',
        default=None,
        help='the directory for saving the SWA model')
    args = parser.parse_args()

    model_dir = args.model_dir
    # model_names = args.model_names

    # models = [torch.load(os.path.join(model_dir, model_name + ".pth")) for model_name in model_names]
    models = [torch.load(model) for model in model_dir]
    model_names = [os.path.splitext(os.path.basename(model))[0] for model in model_dir]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()
    state_dict = models[-1]['state_dict']
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model['state_dict'] = new_state_dict

    merge_name = "-".join(name for name in model_names)
    save_model_name = "ensambled_" + merge_name + ".pth"
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_dir = os.path.join(args.save_dir, save_model_name)
    else:
        save_dir = os.path.join(model_dir, save_model_name)
    torch.save(ref_model, save_dir)
    print('Model is saved at', save_dir)

if __name__ == "__main__":
    main()