_base_ = ['./lsk_s_ema_fpn_1x_dota_le90.py']

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        flip=False)
]