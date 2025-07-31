_base_ = ['../mask-rcnn_r50_fpn_ms-poly-3x_coco.py']

data_root = '/jhcnas4/jh/cytology/CYTO_task/Herlev/'

classes = ('nucleus', 'cytoplasm')  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32),]  # Colors for visualization
)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=2
        ),
        mask_head=dict(
            num_classes=2
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
            ann_file='seg/train.json',
            data_prefix=dict(img='')
        )
    )    
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='seg/val.json',
        data_prefix=dict(img='')
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='seg/test.json',
        data_prefix=dict(img='')
    )
)

val_evaluator = dict(
    ann_file=data_root + 'seg/val.json',
    metric=['bbox','segm']
)
test_evaluator = dict(
    ann_file=data_root + 'seg/test.json',
    metric=['bbox','segm'],
    outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/Herlev_mask_rcnn_r50/Herlev_mask_rcnn_r50_test'
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

work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/Herlev_mask_rcnn_r50'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/mask_rcnn/my/Herlev_mask_rcnn_r50.py > Herlev_mask_rcnn_r50.log 2>&1 &

