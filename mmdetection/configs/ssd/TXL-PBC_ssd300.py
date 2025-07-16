_base_ = './ssd300_coco.py'

data_root = '/jhcnas4/jh/cytology/CYTO_task/TXL-PBC/'

# Define your TXL-PBC dataset classes - replace these with your actual class names
classes = ('RBC', 'WBC', 'Platelets',)  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0),]  # Colors for visualization
)
model = dict(
    bbox_head=dict(
        num_classes=3))

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'))))

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img='val/')))

test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='test/')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox')

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4))

train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)

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

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth'
work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/TXL-PBC_ssd300'
# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/ssd/TXL-PBC_ssd300.py >> TXL-PBC_ssd300.log 2>&1 &