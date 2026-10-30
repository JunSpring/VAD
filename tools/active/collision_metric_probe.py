"""Probe the VAD/ST-P3 collision metric: perturb the GT ego trajectory by mm and watch the
collision flag flip (rasterisation artefact, see docs/activead_ed10.md \u00a77).
Run inside the container: python tools/active/collision_metric_probe.py
"""
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/junspring/VAD')
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # registers
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
cfg = Config.fromfile('projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py')
cfg.data.test.test_mode = True; cfg.data.test.pop('samples_per_gpu', None)
ds = build_dataset(cfg.data.test)
scenes = {s['token']: s['name'] for s in json.load(open('data/nuscenes/v1.0-trainval/scene.json'))}
pm = PlanningMetric()
def run(scene_name, frame_idx, eps_list):
    idx = next(i for i, inf in enumerate(ds.data_infos) if scenes[inf['scene_token']] == scene_name and inf['frame_idx'] == frame_idx)
    d = ds[idx]
    gt_bbox = d['gt_bboxes_3d'][0].data; gt_attr = d['gt_attr_labels'][0].data
    ego_fut = d['ego_fut_trajs'][0].data.reshape(-1, 2).cumsum(dim=-2)   # (6,2) GT, lidar frame
    seg, ped = pm.get_label(gt_bbox, gt_attr.unsqueeze(0)); occ = torch.logical_or(seg, ped)
    gt_coll = pm.evaluate_single_coll(ego_fut, occ[0], input_gt=True).numpy()
    print(f'{scene_name} f{frame_idx}: GT traj (m) = {ego_fut.numpy().round(3).tolist()[:2]}...  GT collides per step: {gt_coll.astype(int).tolist()}')
    # nearest vehicle box and its yaw
    b = gt_bbox.tensor.numpy(); k = int(np.argmin(np.hypot(b[:,0], b[:,1])))
    print(f'   nearest box: center=({b[k,0]:.2f},{b[k,1]:.2f}) w={b[k,3]:.2f} l={b[k,4]:.2f} yaw={b[k,6]:.2f}')
    for ex, ey in eps_list:
        tr = ego_fut.clone(); tr[:, 0] += ex; tr[:, 1] += ey
        oc, obc = pm.evaluate_coll(tr[None], ego_fut[None], occ)
        print(f'   perturb (dx={ex:+.3f}, dy={ey:+.3f}) m -> obj_box_col per step {obc.numpy().astype(int).tolist()}  (1s metric = {obc[:2].mean():.2f})')
eps = [(0,0),(0.001,0),(-0.001,0),(0,0.001),(0,-0.001),(0.01,0),(-0.01,0),(0,0.01),(0,-0.01),(0.05,0.05),(-0.05,-0.05)]
run('scene-0344', 3, eps)
run('scene-0344', 20, eps)
run('scene-0345', 2, eps)
