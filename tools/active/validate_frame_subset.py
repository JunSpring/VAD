import sys, numpy as np, torch, random
sys.path.insert(0, '/home/junspring/VAD')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin
cfg = Config.fromfile('projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py')
for name in ('ed10', 'frand10_s0'):
    c = cfg.data.train.copy(); c.ann_file = 'data/nuscenes/vad_nuscenes_infos_temporal_train_%s.pkl' % name
    ds = build_dataset(c)
    n_ctx = sum(1 for i in ds.data_infos if i.get('context_only'))
    print(f'[{name}] infos={len(ds.data_infos)} context_only={n_ctx} len(ds)={len(ds)} flag={len(ds.flag)}')
    # pick a target whose 3 predecessors are present in the pkl
    tgt = None
    for k, i in enumerate(ds.target_idx):
        if i >= 3 and all(ds.data_infos[i-j]['scene_token'] == ds.data_infos[i]['scene_token'] and
                          ds.data_infos[i-j]['frame_idx'] == ds.data_infos[i]['frame_idx'] - j for j in (1, 2, 3)):
            tgt = k; break
    random.seed(0)
    d = ds[tgt]
    metas = d['img_metas'].data           # dict queue_idx -> meta
    tok2f = {i['token']: i['frame_idx'] for i in ds.data_infos}
    fidx = [tok2f[metas[q]['sample_idx']] for q in sorted(metas)]
    sc = {metas[q]['scene_token'] for q in metas}
    cur = ds.data_infos[ds.target_idx[tgt]]
    print(f'   target frame_idx={cur["frame_idx"]} context_only={cur.get("context_only", False)} -> queue frame_idx={fidx}, one scene={len(sc)==1}, '
          f'img={tuple(d["img"].data.shape)}')
    # make sure no context frame can ever be returned as a target
    assert all(not ds.data_infos[i].get('context_only', False) for i in ds.target_idx)
    # a target at scene start (no predecessors) must still work (duplicated history, like normal training)
    first = next(k for k, i in enumerate(ds.target_idx) if ds.data_infos[i]['frame_idx'] <= 1)
    d2 = ds[first]; m2 = d2['img_metas'].data
    print(f'   scene-start target frame_idx={ds.data_infos[ds.target_idx[first]]["frame_idx"]} -> queue {[tok2f[m2[q]["sample_idx"]] for q in sorted(m2)]} ok')
print('VALIDATION OK')
