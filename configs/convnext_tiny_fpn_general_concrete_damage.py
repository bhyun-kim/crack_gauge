_base_ = [
    '_base_/fpn_convnext-v2-atto.py', 
    '../mmsegmentation/configs/_base_/default_runtime.py', 
    '../mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]

crop_size = (1024, 1024)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=5))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

dataset_type = 'ConcreteDamageDataset'
 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize', 
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

label_map = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:3,
    6:1,
    7:4,
    8:4,
}

general_concrete_damage_root='/home/user/#data/2023.03.25 General Concrete Damage Training/v0.1.1'
general_concrete_damage = dict(
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.5,
        dataset=dict(
            type=dataset_type,
            data_root=general_concrete_damage_root,
            data_prefix=dict(
                img_path='leftImg8bit/train',
                seg_map_path='gtFine/train'
            ),
            label_map=label_map,
            pipeline=train_pipeline,
            )),
    val=dict(
        type=dataset_type,
        data_root=general_concrete_damage_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        label_map=label_map,
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        data_root=general_concrete_damage_root,
        data_prefix=dict(
            img_path='leftImg8bit/test',
            seg_map_path='gtFine/test'
        ),
        label_map=label_map,
        pipeline=test_pipeline,
        )
)


kcqr_concrete_damage_root='/media/user/WDS/#data/2022onsiteimgsplit'
kcqr_concrete_damage = dict(
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.5,
        dataset=dict(
            type=dataset_type,
            data_root=kcqr_concrete_damage_root,
            data_prefix=dict(
                img_path='leftImg8bit/train',
                seg_map_path='gtFine/train'
            ),
            label_map=label_map,
            pipeline=train_pipeline,
            )),
    val=dict(
        type=dataset_type,
        data_root=kcqr_concrete_damage_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        label_map=label_map,
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        data_root=kcqr_concrete_damage_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        label_map=label_map,
        pipeline=test_pipeline,
        )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            general_concrete_damage['train'], 
            kcqr_concrete_damage['train']
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            general_concrete_damage['val'],
            kcqr_concrete_damage['val']
        ]))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            general_concrete_damage['test'],
            kcqr_concrete_damage['test']
        ]))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

work_dir = '/media/user/WDS/#checkpoints/023. KCQR/convnext_tiny_fpn_kcqr_concrete_area_damage'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500)
    )
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)


