_base_ = [
    '../_base_/datasets/deep_crack.py',
    '../_base_/models/cgnet.py', 
    '../_base_/segmentor_runtime.py'
]

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_512x1024_60k_cityscapes/cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth'

crop_size = (2560, 2560)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(class_weight=[5., 10.])))

# optimizer
optimizer = dict(type='Adam', lr=0.001, eps=1e-08, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        by_epoch=False,
        begin=0,
        end=20000)
]

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

load_from = pretrained
