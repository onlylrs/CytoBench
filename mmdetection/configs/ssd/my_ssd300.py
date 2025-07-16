_base_ = './ssd300_coco.py'
dataset_type = 'CocoDataset'
backend_args = None
data_root = '/jhcnas4/jh/cytology/CYTO_task/TXL-PBC/'

metainfo = {
    'classes': ('WBC', 'RBC', 'Platelets'),  # Update these with your actual class names
    'palette': [(220, 20, 60), (119, 11, 32), (0, 255, 0)]  # Colors for visualization
}

model = dict(
    bbox_head=dict(
        num_classes=3))

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
            backend_args=backend_args)))
val_dataloader = dict(batch_size=8, dataset=dict(
    data_root=data_root,
    metainfo=metainfo,
    ann_file='val.json',
    data_prefix=dict(img='val/')
))
test_dataloader = dict(batch_size=8, dataset=dict(
    data_root=data_root,
    metainfo=metainfo,
    ann_file='test.json',
    data_prefix=dict(img='test/')
))

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
work_dir = 'work_dirs/TXL-PBC_ssd300'
# CUDA_VISIBLE_DEVICES=3 python tools/train.py configs/ssd/my_ssd300.py