_base_ = [
    './ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'
]

data_root = '/jhcnas3/Cervical/CervicalData_NEW/Processed_Data/PATCH_DATA/Coco/coco_all/'

# Define your CCS dataset classes - replace these with your actual class names
classes = ('ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC',)  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),]  # Colors for visualization
)

model = dict(
    bbox_head=dict(
        num_classes=6))

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1)
input_size = 320
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    batch_sampler=None,
    dataset=dict(
        dataset=dict(
            _delete_=True,
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline
    )
)

# training schedule
max_epochs = 120
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        eta_min=0)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=4.0e-5))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox'
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox',
    outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/CCS_ssdlite_mobilenetv2/CCS_ssdlite_mobilenetv2_test'
)
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=5,
    save_best='coco/bbox_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=1,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
auto_scale_lr = dict(base_batch_size=192)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'
work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/CCS_ssdlite_mobilenetv2'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/ssd/CCS_ssdlite_mobilenetv2.py > logs/CCS_ssdlite_mobilenetv2.log 2>&1 &