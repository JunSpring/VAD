"""Planning-metric breakdown by scenario (ActiveAD Table 3 style).

Input: the per-sample `metric_record.json` written by tools/test.py --json_dir, the
val info pkl (for per-scene driving commands) and the nuScenes scene.json (weather).

    python tools/active/eval_breakdown.py \
        --metric-json work_dirs/<run>/eval_ep20/metric_record.json \
        --val-pkl data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
        --nusc-dir data/nuscenes/v1.0-trainval

Reported per group, over samples with fut_valid_flag == 1:
  L2 (m)  = mean over {1s,2s,3s} of plan_L2_{t}s
  Col (%) = 100 * mean over {1s,2s,3s} of plan_obj_box_col_{t}s
Paper (10% ED, VAD-Tiny): All 0.83 / 0.41-0.43; Random 0.86 / 0.67; 100% data 0.70 / 0.25.
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_attrs import COMMAND_NAMES, load_scenes, scene_attributes  # noqa: E402

GROUPS = [
    ('Day', lambda a: a['weather'][0] == 'D'),
    ('Night', lambda a: a['weather'][0] == 'N'),
    ('Sunny', lambda a: a['weather'][1] == 'S'),
    ('Rainy', lambda a: a['weather'][1] == 'R'),
    ('Go Straight', lambda a: a['command'] == 'S'),
    ('Turn Left', lambda a: a['command'] == 'L'),
    ('Turn Right', lambda a: a['command'] == 'R'),
    ('Overtake', lambda a: a['command'] == 'O'),
    ('All', lambda a: True),
]
HORIZONS = (1, 2, 3)


def summarize(records):
    recs = [r for r in records if r.get('fut_valid_flag')]
    if not recs:
        return dict(n=0)
    out = dict(n=len(recs))
    for t in HORIZONS:
        out['L2_{}s'.format(t)] = float(np.mean([r['plan_L2_{}s'.format(t)] for r in recs]))
        out['Col_{}s'.format(t)] = 100 * float(np.mean([r['plan_obj_box_col_{}s'.format(t)] for r in recs]))
        out['ColObj_{}s'.format(t)] = 100 * float(np.mean([r['plan_obj_col_{}s'.format(t)] for r in recs]))
    out['L2_avg'] = float(np.mean([out['L2_{}s'.format(t)] for t in HORIZONS]))
    out['Col_avg'] = float(np.mean([out['Col_{}s'.format(t)] for t in HORIZONS]))
    out['ColObj_avg'] = float(np.mean([out['ColObj_{}s'.format(t)] for t in HORIZONS]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--metric-json', required=True)
    p.add_argument('--val-pkl', default='data/nuscenes/vad_nuscenes_infos_temporal_val.pkl')
    p.add_argument('--nusc-dir', default='data/nuscenes/v1.0-trainval')
    p.add_argument('--tau-c', type=int, default=4)
    p.add_argument('--out', default=None, help='breakdown json (default: next to metric json)')
    args = p.parse_args()

    with open(args.metric_json) as f:
        records = json.load(f)
    val = mmcv.load(args.val_pkl)
    attrs = scene_attributes(val['infos'], load_scenes(args.nusc_dir), tau_c=args.tau_c)

    by_scene = {}
    for r in records:
        by_scene.setdefault(r['scene_token'], []).append(r)
    missing = set(by_scene) - set(attrs)
    if missing:
        print('WARNING: {} scene tokens in metric json not in val pkl'.format(len(missing)))

    result = {}
    for name, pred in GROUPS:
        recs = [r for tok, rs in by_scene.items() if tok in attrs and pred(attrs[tok]) for r in rs]
        n_scenes = sum(1 for tok in by_scene if tok in attrs and pred(attrs[tok]))
        result[name] = dict(summarize(recs), n_scenes=n_scenes)

    print('\nscene categories in val: weather', dict(sorted(
        __import__('collections').Counter(a['weather'] for a in attrs.values()).items())),
        ' command', {COMMAND_NAMES[k]: v for k, v in sorted(
            __import__('collections').Counter(a['command'] for a in attrs.values()).items())})
    print('\n{:12s} {:>7s} {:>7s} | {:>6s} {:>6s} {:>6s} {:>7s} | {:>6s} {:>6s} {:>6s} {:>7s}'.format(
        'group', 'scenes', 'frames', 'L2 1s', 'L2 2s', 'L2 3s', 'L2 avg', 'Col1s', 'Col2s', 'Col3s', 'Col avg'))
    for name, _ in GROUPS:
        r = result[name]
        if r['n'] == 0:
            print('{:12s} {:7d} {:7d} | (no valid samples)'.format(name, r['n_scenes'], 0))
            continue
        print('{:12s} {:7d} {:7d} | {:6.3f} {:6.3f} {:6.3f} {:7.3f} | {:6.2f} {:6.2f} {:6.2f} {:7.2f}'.format(
            name, r['n_scenes'], r['n'], r['L2_1s'], r['L2_2s'], r['L2_3s'], r['L2_avg'],
            r['Col_1s'], r['Col_2s'], r['Col_3s'], r['Col_avg']))
    print('\n(Col = plan_obj_box_col, the collision rate VAD reports; ColObj = plan_obj_col)')
    print('Paper 10% ED VAD-Tiny: All L2 0.83 / Col 0.41-0.43 ; Random 0.86 / 0.67 ; full data 0.70 / 0.25')

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.metric_json)), 'breakdown.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
