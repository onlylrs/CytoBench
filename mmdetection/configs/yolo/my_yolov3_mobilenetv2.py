_base_ = './yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'
data_root = '/jhcnas4/jh/cytology/CYTO_task/CBC/'
# data_root = '/jhcnas3/Cervical/CervicalData_NEW/Processed_Data/PATCH_DATA/Coco/coco_all/'

# Define your CRIC dataset classes - replace these with your actual class names
classes = ('RBC', 'WBC', 'Platelets',)  # Update these with your actual class names
# metainfo = dict(
#     classes=classes,
#     palette=[(220, 20, 60), (119, 11, 32), (0, 255, 0),]  # Colors for visualization
# )

model = dict(
    bbox_head=dict(
        num_classes=3))

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/')))


val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
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

# CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py configs/yolo/my_yolov3_mobilenetv2.py >> my_yolov3_mobilenetv2.log 2>&1 &

