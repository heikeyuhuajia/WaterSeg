# dataset settings
dataset_type = 'OurWater'
data_root = '/home/wyuan/cyberpunk/Data/waterDataset'   # 公司数据集存放的路径;
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (360,640)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(360,640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #img_dir='test/img',
        img_dir='test/img_png',             # 测试集图片存放的文件夹名: 图片;
        ann_dir='test/label_gray01_png',    # 测试集图片存放的文件夹名: 标签;
        pipeline=test_pipeline))
