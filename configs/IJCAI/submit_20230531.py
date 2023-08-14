_base_ = ['../lsknet/lsk_s_ema_fpn_1x_dota_le90.py']

dataset_type = 'IJCAIDataset'
data_root = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2'

angle_version = 'le90'
gpu_number = 8

Pretrained = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/backbone_pretrain/20230524-095324-lsk_m-224/model_best.pth.tar"
# load_from = '/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/self_pretrain_merge_dota_lsknet_m_224.pth'
# Pretrained = "/mnt/cfs_bj/nihao/data/ICJAI2023/output/20230521-161033-lsk_s-224/checkpoint-84.pth.tar"
# load_from = "/mnt/cfs_bj/nihao/data/ICJAI2023/pretrain_models/self_pretrain_merge_dota_lsknet_s_224.pth"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint=Pretrained),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    # backbone=dict(
    #     type='LSKNet',
    #     embed_dims=[64, 128, 320, 512],
    #     drop_rate=0.1,
    #     drop_path_rate=0.1,
    #     depths=[2,2,4,2],
    #     init_cfg=dict(type='Pretrained', checkpoint=Pretrained),
    #     norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedGenericRoIExtractor',
            aggregation='sum',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False,
            ),
            post_cfg=dict(
                type='GeneralizedAttention',
                in_channels=256,
                spatial_range=-1,
                num_heads=6,
                attention_type='0100',
                kv_stride=2)),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            reg_decoded_bbox=True,
            loss_bbox=dict(
                    _delete_=True,
                    type='GDLoss_v1',
                    loss_type='kld',
                    fun='log1p',
                    tau=1,
                    loss_weight=1.0)
            )),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RResize', img_scale=(1024, 1024)),
    # dict(
    #     type="RMosaic",
    #     img_scale=(1024, 1024),
    #     prob=0.5,
    #     pad_val=114.0
    # ),
    dict(type='RResize', img_scale=[(4096, 512), (4096, 1024)]),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        # rect_classes=[9, 11],
        rect_classes=[0,1,2,3,4,5,6,7,8],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4096, 800),
        # img_scale=(4096, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_train = dict(
        type=dataset_type,
        ann_file=data_root + '/train/annotations/',
        img_prefix=data_root + '/train/images/',
        pipeline=train_pipeline
    )
# dataset_train = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root + '/train/annotations/',
#         img_prefix=data_root + '/train/images/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#         ],
#         filter_empty_gt=False,
#         ),
#     pipeline=train_pipeline
#     )


# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=10,
#     train=dict(
#         type='ConcatDataset',
#         datasets=[dataset_ori]
#     ),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + '/eval/annotations/',
#         img_prefix=data_root + '/eval/images/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file="/mnt/cfs_bj/nihao/data/ICJAI2023/test1_phase1/images/",
#         img_prefix="/mnt/cfs_bj/nihao/data/ICJAI2023/test1_phase1/images/",
#         pipeline=test_pipeline))
# local
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=10,
    train=dict(
        type='ConcatDataset',
        datasets=[dataset_train, dataset_train, dataset_train, dataset_train]
        # datasets=[dataset_train, dataset_train, dataset_train, dataset_train, dataset_train]
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/eval/annotations',
        img_prefix=data_root + '/eval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/all_k_folds/fold_3/annotations',
        img_prefix=data_root + '/all_k_folds/fold_3/images',
        # ann_file="/mnt/cfs_bj/nihao/data/ICJAI2023/test1_phase1/images/",
        # img_prefix="/mnt/cfs_bj/nihao/data/ICJAI2023/test1_phase1/images/",
        pipeline=test_pipeline))

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[24, 33])
# runner = dict(type='EpochBasedRunner', max_epochs=36)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)
checkpoint_config = dict(interval=1)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05)