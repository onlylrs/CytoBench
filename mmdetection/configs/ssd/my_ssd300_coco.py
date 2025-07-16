_base_ = [
    '../_base_/models/ssd300.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco_detection.py'
]

# dataset settings
data_root = '/jhcnas4/jh/cytology/CYTO_task/TXL-PBC/'

metainfo = {
    'classes': ('WBC', 'RBC', 'Platelets'),  # Update these with your actual class names
    'palette': [(220, 20, 60), (119, 11, 32), (0, 255, 0)]  # Colors for visualization
}


# Custom dataset settings for cytology dataset
dataset_type = 'CocoDataset'
backend_args = None

# Dataset metadata for cytology classification  


# Model configuration - override num_classes for our 2-class dataset
model = dict(
    bbox_head=dict(
        num_classes=3
    )
)

# SSD300 specific pipeline settings
input_size = 300
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))
val_dataloader = dict(batch_size=8, dataset=dict(
    data_root=data_root,
    metainfo=metainfo,
    ann_file='val.json',
    data_prefix=dict(img='val/'),
    pipeline=test_pipeline,
))
test_dataloader = dict(batch_size=8, dataset=dict(
    data_root=data_root,
    metainfo=metainfo,
    ann_file='test.json',
    data_prefix=dict(img='test/'),
    pipeline=test_pipeline,
))

# Update evaluators to use custom dataset files
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    outfile_prefix='work_dirs/TXL-PBC_ssd300/TXL-PBC_ssd300_test'
)

# Training configuration - disable validation to avoid SSDHead compatibility issues
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,  # Save checkpoint every epoch
        save_best='coco/bbox_mAP',  # Save best checkpoint based on bbox mAP
        rule='greater',  # Higher bbox mAP is better
        max_keep_ckpts=1  # Keep only the latest 3 checkpoints to save disk space
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth'
work_dir = 'work_dirs/TXL-PBC_ssd300'