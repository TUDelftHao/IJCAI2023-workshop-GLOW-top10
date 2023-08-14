#!/usr/bin/env bash

config_name=$1
CHECKPOINT=$2
GPUS=$3
test_type=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

OUT_DIR=/mnt/cfs_bj/nihao/data/ICJAI2023/output


CONFIG="configs/ICJAI/${config_name}.py"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH


if [[ ${test_type} == "eval" ]]; then
# 评估mAP
python -W ignore -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval mAP \
    --launcher pytorch \
    "${@:5}"
elif [[ ${test_type} == "submit" ]]; then
# 提交文件
python -W ignore -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --format-only \
    --eval-options submission_dir=${OUT_DIR}/k_fold_${config_name}/submit_results \
    --launcher pytorch \
    "${@:5}"
elif [[ ${test_type} == "pkl" ]]; then
# 输出mmdet格式文件
python -W ignore -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --out=${OUT_DIR}/${config_name}/${config_name}.pkl \
    --show-dir=${OUT_DIR}/${config_name}/vis
    "${@:5}"
fi