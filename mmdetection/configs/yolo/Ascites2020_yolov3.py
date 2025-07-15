_base_ = './yolov3_d53_8xb8-ms-416-273e_coco.py'

data_root = '/jhcnas4/jh/cytology/CYTO_task/Ascites2020/det/'

# Define your Ascites2020 dataset classes - replace these with your actual class names
classes = ('Benign_eosinophil_granulocyte', 'Benign_lymphocyte', 'Benign_mesothelial', 'Benign_neutrophil_granulocyte', 'Malignant_Determined', 'Malignant_Suspicious', )  # Update these with your actual class names
metainfo = dict(
    classes=classes,
    palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0),]  # Colors for visualization
)

model = dict(
    bbox_head=dict(
        num_classes=6))

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/')))


val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/')))

val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric='bbox'
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    metric='bbox'
)
train_cfg = dict(max_epochs=273, val_interval=7)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', 
    interval=7,
    save_best='coco/bbox_mAP',  # Save checkpoint with best mAP
    rule='greater',             # Higher mAP is better
    max_keep_ckpts=2,          # Keep only 2 checkpoints (latest + best)
    save_last=True             # Always save the latest checkpoint
))

work_dir = '/jhcnas2/home/jh/CARE/bench/rliuar/Ascites2020_yolov3'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/yolo/Ascites2020_yolov3.py >> Ascites2020_yolov3.log 2>&1 &