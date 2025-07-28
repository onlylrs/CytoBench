_base_ = ['dino-5scale_swin-l_8xb2-12e_coco.py']

data_root = '/jhcnas4/jh/cytology/CYTO_task/TXL-PBC/'

# Define your TXL-PBC dataset classes - replace these with your actual class names
classes = ('RBC', 'WBC', 'Platelet',)  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0),]  # Colors for visualization
)
model = dict(
    bbox_head=dict(
        num_classes=3)
)
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        data_root=data_root,
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        data_root=data_root,
        metainfo=metainfo))
test_dataloader = dict(
    dataset=dict(
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        data_root=data_root,
        metainfo=metainfo))
val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = dict(ann_file=data_root + 'test.json',
outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/TXL-PBC_dino_swinL/TXL-PBC_dino_swinL_test')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=1,
    save_best='coco/bbox_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=1,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))
work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/TXL-PBC_dino_swinL'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/dino/TXL-PBC_dino_swinL.py > logs/TXL-PBC_dino_swinL.log 2>&1 &