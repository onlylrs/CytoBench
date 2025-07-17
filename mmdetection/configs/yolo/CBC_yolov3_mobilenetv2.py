_base_ = './yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'

data_root = '/jhcnas4/jh/cytology/CYTO_task/CBC/'

classes = ('RBC', 'WBC', 'Platelets',)  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0),]  # Colors for visualization
)
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)
model = dict(
    bbox_head=dict(
        num_classes=3))
        
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    dataset=dict(  # RepeatDataset level
        dataset=dict(  # Inner CocoDataset level
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            pipeline=train_pipeline
        )
    )
)


val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,  # remove base dataset definition
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline  # ensure inputs are packed
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

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox'
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox',
    outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/CBC_yolov3_mobilenetv2/CBC_yolov3_mobilenetv2_test'
)
train_cfg = dict(max_epochs=30, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=1,
    save_best='coco/bbox_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=1,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))

work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/CBC_yolov3_mobilenetv2'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/yolo/CBC_yolov3_mobilenetv2.py >> CBC_yolov3_mobilenetv2.log 2>&1 &

