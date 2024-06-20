custom_imports = dict(imports=['sfod'], allow_failed_imports=False)
import torchvision.transforms as transforms

gpu = 1
score = 0.7
samples_per_gpu = 2
total_epoch = 1
test_interval = 1
save_interval = 1

classes = ('airplane', 'airport', 'baseballfield','basketballcourt', 'bridge', 'chimney', 'dam', 'Expressway-Service-area','Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship','stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

data_root = 'dataset/DIOR/'
data_root_l = data_root+'JPEGImages'
data_root_u = data_root+'Corruption/JPEGImages-${corrupt}'
ann_file_l = data_root+'ImageSets/Main/train.txt'
ann_file_u = data_root+'ImageSets/Main/val.txt'
ann_file_test = data_root+'ImageSets/Main/test.txt'
ann_subdir = data_root+'Annotations/Oriented Bounding Boxes'

angle_version = 'le90'
# # -------------------------dataset------------------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = (800, 800)

#labeled data pipeline
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg')
         )
]

unsup_pipeline_share = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # generate fake labels for data format compatibility
    # dict(type="LoadEmptyAnnotations", with_bbox=True),
]

unsup_pipeline_weak = [
    dict(type='RResize', img_scale=image_size),
    # dict(type="ExtraAttrs", tag="unsup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],)
]

unsup_pipeline_strong = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg')
         )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type='SemiDIORDataset',
        ann_file=ann_file_l,
        ann_file_u=ann_file_u,
        ann_subdir=ann_subdir,
        pipeline=sup_pipeline, pipeline_u_share=unsup_pipeline_share,
        pipeline_u=unsup_pipeline_weak, pipeline_u_1=unsup_pipeline_strong,
        img_prefix=data_root_l, img_prefix_u=data_root_u,
        classes=classes
    ),
    val=dict(
        type='DIORDataset',
        ann_file=ann_file_test,
        ann_subdir=ann_subdir,
        img_prefix=data_root_u,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='DIORDataset',
        ann_file=ann_file_test,
        ann_subdir=ann_subdir,
        img_prefix=data_root_u,
        classes=classes,
        pipeline=test_pipeline))


evaluation = dict(interval=test_interval, metric='mAP', only_ema=True)
# evaluation = dict(interval=test_interval, metric='mAP')
# # -------------------------schedule------------------------------
learning_rate = 0.02 * samples_per_gpu * gpu / 32
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[total_epoch]
)
runner = dict(type='SemiEpochBasedRunner', max_epochs=total_epoch)

checkpoint_config = dict(interval=save_interval)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# custom_hooks = [
#     dict(type='NumClassCheckHook'),
# ]

custom_hooks = [
    dict(type='SetEpochInfoHook')
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None

load_from = f'baseline/baseline.pth'
workflow = [('train', 1)]

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_dior_cga.py'
# # -------------------------model------------------------------
model = dict(
    type='UnbiasedTeacher',
    ema_config=ema_config,
    ema_ckpt=load_from,
    cfg=dict(
        weight_l = 0, #weight of supervised loss, set 0 for SFOD.
        weight_u = 1, #weight of unsupervised loss
        debug=False,
        score_thr=score,
        use_bbox_reg=False,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))