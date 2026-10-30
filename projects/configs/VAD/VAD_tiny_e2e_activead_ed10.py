# VAD-Tiny trained on the ActiveAD Ego-Diversity 10% cold-start split
# (70 of the 700 nuScenes train scenes, gamma=0.5, tau_c=4).
#
# Paper settings (ActiveAD, CVPR 2024, Sec. 5.1): VAD-Tiny with default
# hyper-parameters, AdamW, lr 2e-4, weight decay 0.01, CosineAnnealing,
# 20 epochs, trained from scratch (ImageNet ResNet50 init).
#
# This host has a single RTX 4090. VAD's defaults assume 8 GPUs x 1 sample, so
# the effective batch of 8 is emulated with gradient accumulation (x8) while
# keeping lr=2e-4. Everything else is inherited from VAD_tiny_e2e.py.
_base_ = ['./VAD_tiny_e2e.py']

total_epochs = 20  # ActiveAD: 20 epochs (VAD default: 60)
data_root = 'data/nuscenes/'

# mmdet_train.py asserts cfg.total_epochs == cfg.runner.max_epochs and the head
# uses tot_epoch for its loss schedule, so all three must change together.
model = dict(pts_bbox_head=dict(tot_epoch=total_epochs))

data = dict(
    workers_per_gpu=8,
    train=dict(ann_file=data_root + 'vad_nuscenes_infos_temporal_train_ed10.pkl'))

# 1 GPU x bs1 x 8 accumulated iters == 8 GPUs x bs1 (VAD default).
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=35, norm_type=2))

# mmcv's LrUpdaterHook counts dataloader iterations, so VAD's 500 warmup
# optimizer steps (batch 8) become 500 x 8 = 4000 iterations here.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
