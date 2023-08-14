#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7

GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# config_name="lsk_s_ema_fpn_1x_dota_le90_k_fold"
# config_name="submit_20230521"
# config_name="submit_20230530"
# config_name="submit_20230531"
config_name="submit_20230605"


CONFIG="configs/ICJAI/${config_name}.py"


# base_dir=/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/train_k_folds
base_dir=/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/all_k_folds/

# echo "ROUND 1"
# python -W ignore -m torch.distributed.launch \
#             --nnodes=$NNODES \
#             --node_rank=$NODE_RANK \
#             --master_addr=$MASTER_ADDR \
#             --nproc_per_node=$GPUS \
#             --master_port=$PORT \
#             ./tools/train.py \
#             $CONFIG \
#             --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_1/annotations" \
#                         data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_1/images" \
#                         data.train.datasets.1.dataset.ann_file="${base_dir}/fold_2/annotations" \
#                         data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_2/images" \
#                         data.train.datasets.2.dataset.ann_file="${base_dir}/fold_3/annotations" \
#                         data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_3/images" \
#                         data.train.datasets.3.dataset.ann_file="${base_dir}/fold_4/annotations" \
#                         data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_4/images" \
#                         data.val.ann_file="${base_dir}/fold_0/annotations" \
#                         data.val.img_prefix="${base_dir}/fold_0/images" \
#             --seed 0 \
#             --launcher pytorch ${@:3} \
#             --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_0/"

echo "ROUND 2"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_0/annotations" \
                        data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_0/images" \
                        data.train.datasets.1.dataset.ann_file="${base_dir}/fold_2/annotations" \
                        data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_2/images" \
                        data.train.datasets.2.dataset.ann_file="${base_dir}/fold_3/annotations" \
                        data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_3/images" \
                        data.train.datasets.3.dataset.ann_file="${base_dir}/fold_4/annotations" \
                        data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_4/images" \
                        data.val.ann_file="${base_dir}/fold_1/annotations" \
                        data.val.img_prefix="${base_dir}/fold_1/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_1/"

echo "ROUND 3"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_0/annotations" \
                        data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_0/images" \
                        data.train.datasets.1.dataset.ann_file="${base_dir}/fold_1/annotations" \
                        data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_1/images" \
                        data.train.datasets.2.dataset.ann_file="${base_dir}/fold_3/annotations" \
                        data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_3/images" \
                        data.train.datasets.3.dataset.ann_file="${base_dir}/fold_4/annotations" \
                        data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_4/images" \
                        data.val.ann_file="${base_dir}/fold_2/annotations" \
                        data.val.img_prefix="${base_dir}/fold_2/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_2/"


echo "ROUND 4"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_0/annotations" \
                        data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_0/images" \
                        data.train.datasets.1.dataset.ann_file="${base_dir}/fold_1/annotations" \
                        data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_1/images" \
                        data.train.datasets.2.dataset.ann_file="${base_dir}/fold_2/annotations" \
                        data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_2/images" \
                        data.train.datasets.3.dataset.ann_file="${base_dir}/fold_4/annotations" \
                        data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_4/images" \
                        data.val.ann_file="${base_dir}/fold_3/annotations" \
                        data.val.img_prefix="${base_dir}/fold_3/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_3/"

echo "ROUND 5"
python -W ignore -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            ./tools/train.py \
            $CONFIG \
            --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_0/annotations" \
                        data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_0/images" \
                        data.train.datasets.1.dataset.ann_file="${base_dir}/fold_1/annotations" \
                        data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_1/images" \
                        data.train.datasets.2.dataset.ann_file="${base_dir}/fold_2/annotations" \
                        data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_2/images" \
                        data.train.datasets.3.dataset.ann_file="${base_dir}/fold_3/annotations" \
                        data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_3/images" \
                        data.val.ann_file="${base_dir}/fold_4/annotations" \
                        data.val.img_prefix="${base_dir}/fold_4/images" \
            --seed 0 \
            --launcher pytorch ${@:3} \
            --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_4/"



# echo "ROUND complete"
# python -W ignore -m torch.distributed.launch \
#             --nnodes=$NNODES \
#             --node_rank=$NODE_RANK \
#             --master_addr=$MASTER_ADDR \
#             --nproc_per_node=$GPUS \
#             --master_port=$PORT \
#             ./tools/train.py \
#             $CONFIG \
#             --cfg-options data.train.datasets.0.dataset.ann_file="${base_dir}/fold_0/annotations" \
#                         data.train.datasets.0.dataset.img_prefix="${base_dir}/fold_0/images" \
#                         data.train.datasets.1.dataset.ann_file="${base_dir}/fold_1/annotations" \
#                         data.train.datasets.1.dataset.img_prefix="${base_dir}/fold_1/images" \
#                         data.train.datasets.2.dataset.ann_file="${base_dir}/fold_2/annotations" \
#                         data.train.datasets.2.dataset.img_prefix="${base_dir}/fold_2/images" \
#                         data.train.datasets.3.dataset.ann_file="${base_dir}/fold_3/annotations" \
#                         data.train.datasets.3.dataset.img_prefix="${base_dir}/fold_3/images" \
#                         data.train.datasets.4.dataset.ann_file="${base_dir}/fold_4/annotations" \
#                         data.train.datasets.4.dataset.img_prefix="${base_dir}/fold_4/images" \
#                         data.val.ann_file="${base_dir}/fold_4/annotations" \
#                         data.val.img_prefix="${base_dir}/fold_4/images" \
#             --seed 0 \
#             --launcher pytorch ${@:3} \
#             --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/k_fold_${config_name}/fold_all/"

# for i in {0..4}
# do
#     echo "Loop $i"
#     # 执行除循环号和脚本名相同的其他脚本
#     for j in {0..4}
#     do
#         if [ "$i" -ne "$j" ]; then
            
            
#         fi
#     done

# done

