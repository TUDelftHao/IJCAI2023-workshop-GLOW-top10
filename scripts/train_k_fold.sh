#!/usr/bin/env bash


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# export CUDA_VISIBLE_DEVICES=4,5,6,7

GPUS=7
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# config_name="lsk_s_ema_fpn_1x_dota_le90_k_fold"
# config_name="submit_20230521"
# config_name="submit_20230530"
# config_name="submit_20230531"
# config_name="submit_20230531"
# config_name="submit_20230604"
# config_name="submit_20230605"
config_name="submit_20230625"


CONFIG="configs/ICJAI/${config_name}.py"


# base_dir=/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train_k_folds
base_dir=/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_k_folds/

# echo "ROUND 1"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_1_paste_knife-lighter/annotations" \
                        data.train.datasets.0.img_prefix="${base_dir}/fold_1_paste_knife-lighter/images" \
                        data.train.datasets.1.ann_file="${base_dir}/fold_2_paste_knife-lighter/annotations" \
                        data.train.datasets.1.img_prefix="${base_dir}/fold_2_paste_knife-lighter/images" \
                        data.train.datasets.2.ann_file="${base_dir}/fold_3_paste_knife-lighter/annotations" \
                        data.train.datasets.2.img_prefix="${base_dir}/fold_3_paste_knife-lighter/images" \
                        data.train.datasets.3.ann_file="${base_dir}/fold_4_paste_knife-lighter/annotations" \
                        data.train.datasets.3.img_prefix="${base_dir}/fold_4_paste_knife-lighter/images" \
                        data.val.ann_file="${base_dir}/fold_0/annotations" \
                        data.val.img_prefix="${base_dir}/fold_0/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_0/"

# echo "ROUND 2"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_0_paste/annotations" \
                        data.train.datasets.0.img_prefix="${base_dir}/fold_0_paste/images" \
                        data.train.datasets.1.ann_file="${base_dir}/fold_2_paste/annotations" \
                        data.train.datasets.1.img_prefix="${base_dir}/fold_2_paste/images" \
                        data.train.datasets.2.ann_file="${base_dir}/fold_3_paste/annotations" \
                        data.train.datasets.2.img_prefix="${base_dir}/fold_3_paste/images" \
                        data.train.datasets.3.ann_file="${base_dir}/fold_4_paste/annotations" \
                        data.train.datasets.3.img_prefix="${base_dir}/fold_4_paste/images" \
                        data.val.ann_file="${base_dir}/fold_1/annotations" \
                        data.val.img_prefix="${base_dir}/fold_1/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_1/"

# echo "ROUND 3"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_0_paste/annotations" \
                        data.train.datasets.0.img_prefix="${base_dir}/fold_0_paste/images" \
                        data.train.datasets.1.ann_file="${base_dir}/fold_1_paste/annotations" \
                        data.train.datasets.1.img_prefix="${base_dir}/fold_1_paste/images" \
                        data.train.datasets.2.ann_file="${base_dir}/fold_3_paste/annotations" \
                        data.train.datasets.2.img_prefix="${base_dir}/fold_3_paste/images" \
                        data.train.datasets.3.ann_file="${base_dir}/fold_4_paste/annotations" \
                        data.train.datasets.3.img_prefix="${base_dir}/fold_4_paste/images" \
                        data.val.ann_file="${base_dir}/fold_2/annotations" \
                        data.val.img_prefix="${base_dir}/fold_2/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_2/"


# echo "ROUND 4"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_0_paste/annotations" \
                        data.train.datasets.0.img_prefix="${base_dir}/fold_0_paste/images" \
                        data.train.datasets.1.ann_file="${base_dir}/fold_1_paste/annotations" \
                        data.train.datasets.1.img_prefix="${base_dir}/fold_1_paste/images" \
                        data.train.datasets.2.ann_file="${base_dir}/fold_2_paste/annotations" \
                        data.train.datasets.2.img_prefix="${base_dir}/fold_2_paste/images" \
                        data.train.datasets.3.ann_file="${base_dir}/fold_4_paste/annotations" \
                        data.train.datasets.3.img_prefix="${base_dir}/fold_4_paste/images" \
                        data.val.ann_file="${base_dir}/fold_3/annotations" \
                        data.val.img_prefix="${base_dir}/fold_3/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_3/"

# echo "ROUND 5"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_0_paste/annotations" \
                        data.train.datasets.0.img_prefix="${base_dir}/fold_0_paste/images" \
                        data.train.datasets.1.ann_file="${base_dir}/fold_1_paste/annotations" \
                        data.train.datasets.1.img_prefix="${base_dir}/fold_1_paste/images" \
                        data.train.datasets.2.ann_file="${base_dir}/fold_2_paste/annotations" \
                        data.train.datasets.2.img_prefix="${base_dir}/fold_2_paste/images" \
                        data.train.datasets.3.ann_file="${base_dir}/fold_3_paste/annotations" \
                        data.train.datasets.3.img_prefix="${base_dir}/fold_3_paste/images" \
                        data.val.ann_file="${base_dir}/fold_4/annotations" \
                        data.val.img_prefix="${base_dir}/fold_4/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_4/"


echo "ROUND complete"
# python -W ignore -m torch.distributed.launch \
#             --nnodes=$NNODES \
#             --node_rank=$NODE_RANK \
#             --master_addr=$MASTER_ADDR \
#             --nproc_per_node=$GPUS \
#             --master_port=$PORT \
#             ./tools/train.py \
#             $CONFIG \
#             --cfg-options data.train.datasets.0.ann_file="${base_dir}/fold_0/annotations" \
#                         data.train.datasets.0.img_prefix="${base_dir}/fold_0/images" \
#                         data.train.datasets.1.ann_file="${base_dir}/fold_1/annotations" \
#                         data.train.datasets.1.img_prefix="${base_dir}/fold_1/images" \
#                         data.train.datasets.2.ann_file="${base_dir}/fold_2/annotations" \
#                         data.train.datasets.2.img_prefix="${base_dir}/fold_2/images" \
#                         data.train.datasets.3.ann_file="${base_dir}/fold_3/annotations" \
#                         data.train.datasets.3.img_prefix="${base_dir}/fold_3/images" \
#                         data.train.datasets.4.ann_file="${base_dir}/fold_4/annotations" \
#                         data.train.datasets.4.img_prefix="${base_dir}/fold_4/images" \
#                         data.val.ann_file="${base_dir}/fold_4/annotations" \
#                         data.val.img_prefix="${base_dir}/fold_4/images" \
#             --seed 0 \
#             --launcher pytorch ${@:3} \
#             --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_all/"


