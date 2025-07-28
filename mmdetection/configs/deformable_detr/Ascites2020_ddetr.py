_base_ = 'deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'

data_root = '/jhcnas4/jh/cytology/CYTO_task/Ascites2020/det/'

# Define your Ascites2020 dataset classes - replace these with your actual class names
classes = ('cell', )  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60),]  # Colors for visualization
)
model = dict(
    bbox_head=dict(
        num_classes=1)
)

max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
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
outfile_prefix='/jhcnas2/home/jh/CARE/bench/rliuar/Ascites2020_ddetr/Ascites2020_ddetr_test')
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=1,
    save_best='coco/bbox_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=1,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))
work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/Ascites2020_ddetr'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/deformable_detr/Ascites2020_ddetr.py > logs/Ascites2020_ddetr.log 2>&1 &
