_base_ = ['./cascade-mask-rcnn_r50_fpn_ms-3x_coco.py']
data_root = '/jhcnas4/jh/cytology/CYTO_task/CellaVision/'

classes = ('nucleus', 'cytoplasm')  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32),]  # Colors for visualization
)
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(
            num_classes=2,
        )
    )
)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/')
        )
    )    
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/')
    )
)

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric=['bbox','segm']
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric=['bbox','segm'],
    outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/CellaVision_cascade_r50/CellaVision_cascade_r50_test'
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=1,
    save_best='coco/segm_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=1,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))

work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/CellaVision_cascade_r50'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824.pth'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/cascade_rcnn/CellaVision_cascade_r50.py > logs/CellaVision_cascade_r50.log 2>&1 &