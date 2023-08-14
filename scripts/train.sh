#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# config_name="lsk_s_ema_fpn_1x_dota_le90_6x"
# config_name="lsk_s_ema_fpn_1x_dota_le90_classbalance"
# config_name="lsk_s_ema_fpn_1x_dota_le90_mosaic"
# config_name="lsk_s_ema_fpn_1x_dota_le90_data_aug"
# config_name="lsk_s_ema_fpn_1x_dota_le90_refpn"
# config_name="lsk_s_ema_fpn_1x_dota_le90_data_aug_6x"
# config_name="lsk_s_ema_fpn_1x_dota_le90_kfiou"
# config_name="lsk_s_ema_fpn_1x_dota_le90_swa"
# config_name="lsk_s_ema_fpn_1x_dota_le90_conv2d"
# config_name="lsk_s_ema_fpn_1x_dota_le90_nasfpn"
# config_name="lsk_s_ema_fpn_1x_dota_le90_nopretrain"
# config_name="lsk_s_ema_fpn_1x_dota_le90_focalloss"
# config_name="lsk_s_ema_fpn_1x_dota_le90_data_aug_6x_pretrain"
# config_name="lsk_s_ema_fpn_1x_dota_le90_custom_eval"
# config_name="lsk_s_ema_fpn_1x_dota_le90_custom_eval_conv2d"
# config_name="lsk_s_ema_fpn_1x_dota_le90_custom_eval_nasfpn"
# config_name="lsk_s_ema_fpn_1x_dota_le90_mosaic_crop"
# config_name="lsk_s_ema_fpn_1x_dota_le90_hrfpn"
# config_name="lsk_s_ema_fpn_1x_dota_le90_hrfpn_bfp"
# config_name="lsk_s_ema_fpn_1x_dota_le90_allroi"
# config_name="lsk_s_ema_fpn_1x_dota_le90_occ_module"
# config_name="lsk_s_ema_fpn_1x_dota_le90_hrfpn_bfp_mosaic_ewa"
# config_name="lsk_s_ema_fpn_1x_dota_le90_hrfpn_bfp_ewa"
# config_name="lsk_s_ema_fpn_1x_dota_le90_data_aug_6x_scale_aug"
# config_name="lsk_s_ema_fpn_1x_dota_le90"
# config_name="lsk_s_ema_fpn_1x_dota_le90_pan"
# config_name="lsk_s_ema_fpn_1x_dota_le90_bfp"
# config_name="lsk_s_ema_fpn_1x_dota_le90_fpn_dyhead"
# config_name="lsk_s_ema_fpn_1x_dota_le90_caraf"
# config_name="lsk_s_ema_fpn_1x_dota_le90_pan_bfp"
# config_name="lsk_s_ema_fpn_1x_dota_le90_hrfpn"
# config_name="lsk_s_ema_fpn_1x_dota_le90_multiscale"
# config_name="lsk_s_ema_fpn_1x_dota_oc"
# config_name="lsk_s_ema_fpn_1x_dota_le90_focalloss"

# config_name="lsk_s_ema_fpn_1x_dota_le90_crop" BUG

# config_name="lsk_s_ema_fpn_1x_dota_le90_BalancedL1Loss"
# config_name="lsk_s_ema_fpn_1x_dota_le90_iouloss"
# config_name="lsk_s_ema_fpn_1x_dota_le90_libra"
# config_name="lsk_s_ema_fpn_1x_dota_le90_lr2x"

# config_name="lsk_s_ema_fpn_1x_dota_le90_backbone_trained224"
# config_name="lsk_s_ema_fpn_1x_dota_le90_merge_selfpretrainbk_dota"
# config_name="lsk_s_ema_fpn_1x_dota_le90_allroi"
# config_name="submit_20230521"
config_name="lsk_m_ema_fpn_1x_dota_le90"

CONFIG="configs/ICJAI/${config_name}.py"
python -W ignore -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3} \
    --work-dir "/mnt/cfs_bj/nihao/data/ICJAI2023/output/${config_name}/"
