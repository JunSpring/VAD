"""ActiveAD Ego-Diversity initial (cold-start) selection on nuScenes (Sec. 4.1, Alg. 1).

Works on the VAD info pkl produced by tools/data_converter/vad_nuscenes_converter.py so
that the selection is computed from exactly the labels the model trains on.

    # sanity check against the paper's Table 4 "Complete" row
    python tools/active/ego_diversity_select.py check \
        --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
        --nusc-dir data/nuscenes/v1.0-trainval

    # select 10% (70 scenes), gamma=0.5, tau_c=4
    python tools/active/ego_diversity_select.py select \
        --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
        --nusc-dir data/nuscenes/v1.0-trainval --gamma 0.5 --budget 70 \
        --out tools/active/splits/ed10_gamma0.5.json

    # write the filtered train pkl used by the config
    python tools/active/ego_diversity_select.py filter \
        --train-pkl data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
        --scenes tools/active/splits/ed10_gamma0.5.json \
        --out /media/junspring/JunBoom/dataset/nuScenes/vad_infos/vad_nuscenes_infos_temporal_train_ed10.pkl

Decisions the paper leaves open (all deterministic, all recorded in the split json):
* rounding of n0 * P_s: largest-remainder per level; the second level is distributed
  from the *integer* first-level allocation (ties -> larger cell, then fixed order).
* "regular intervals" over the speed-sorted cell: centred strata,
  idx = floor((arange(k) + 0.5) * n / k).
* speed: see scene_attrs.frame_speed (CAN v0, pose fallback).
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_attrs import (COMMAND_CELLS, WEATHER_CELLS, load_scenes,  # noqa: E402
                         scene_attributes)

PAPER_COMPLETE = dict(weather=dict(DS=491, DR=125, NS=71, NR=13),
                      command=dict(S=423, R=132, L=112, O=33))
PAPER_GAMMA05 = dict(weather=dict(DS=34, DR=17, NS=13, NR=6),
                     command=dict(S=27, R=17, L=16, O=10))


def count_table(attrs):
    table = {x: {y: 0 for y in COMMAND_CELLS} for x in WEATHER_CELLS}
    for a in attrs.values():
        table[a['weather']][a['command']] += 1
    return table


def marginals(table):
    return dict(weather={x: sum(table[x].values()) for x in WEATHER_CELLS},
                command={y: sum(table[x][y] for x in WEATHER_CELLS) for y in COMMAND_CELLS})


def largest_remainder(weights, total, caps=None):
    """Integer allocation of `total` proportional to `weights` (largest-remainder method).

    Ties on the remainder go to the larger weight, then to the earlier index.
    `caps` bounds each entry (a cell cannot receive more scenes than it has).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0 or total == 0:
        return [0] * len(w), [0.0] * len(w)
    exact = total * w / w.sum()
    alloc = np.floor(exact).astype(int)
    if caps is not None:
        alloc = np.minimum(alloc, caps)
    remainder = exact - alloc
    order = sorted(range(len(w)), key=lambda i: (-remainder[i], -w[i], i))
    left = total - int(alloc.sum())
    for i in order:
        if left <= 0:
            break
        if caps is not None and alloc[i] >= caps[i]:
            continue
        alloc[i] += 1
        left -= 1
    return [int(a) for a in alloc], [float(e) for e in exact]


def allocate(table, budget, gamma):
    """Eq. (1)/(2): P_x = n_x^g / sum n_z^g ; P_xy = P_x * n_xy^g / sum_z n_xz^g ; n_xy = n0 * P_xy."""
    n_x = [sum(table[x].values()) for x in WEATHER_CELLS]
    w_x = [n ** gamma if n > 0 else 0.0 for n in n_x]
    k_x, exact_x = largest_remainder(w_x, budget, caps=n_x)
    alloc, exact = {}, {}
    for i, x in enumerate(WEATHER_CELLS):
        n_xy = [table[x][y] for y in COMMAND_CELLS]
        w_xy = [n ** gamma if n > 0 else 0.0 for n in n_xy]
        k_xy, exact_xy = largest_remainder(w_xy, k_x[i], caps=n_xy)
        alloc[x] = dict(zip(COMMAND_CELLS, k_xy))
        # unrounded second level uses the unrounded first level (pure Eq. 2)
        s = sum(w_xy)
        exact[x] = {y: (exact_x[i] * w / s if s > 0 else 0.0) for y, w in zip(COMMAND_CELLS, w_xy)}
    return alloc, exact


def regular_interval_indices(n, k):
    if k <= 0:
        return []
    return sorted(set(int(i) for i in np.floor((np.arange(k) + 0.5) * n / k)))


def print_table(table, title):
    print(title)
    print('      ' + ''.join('{:>6s}'.format(y) for y in COMMAND_CELLS) + '   sum')
    for x in WEATHER_CELLS:
        row = table[x]
        vals = [row[y] for y in COMMAND_CELLS]
        fmt = '{:6.1f}' if any(isinstance(v, float) for v in vals) else '{:6d}'
        print('  {:4s}'.format(x) + ''.join(fmt.format(v) for v in vals) + fmt.format(sum(vals)))


def build_attrs(args):
    data = mmcv.load(args.train_pkl)
    scenes = load_scenes(args.nusc_dir)
    attrs = scene_attributes(data['infos'], scenes, tau_c=args.tau_c, speed_source=args.speed_source)
    return data, attrs


def cmd_check(args):
    _, attrs = build_attrs(args)
    table = count_table(attrs)
    marg = marginals(table)
    print_table(table, '\n#scenes per (weather-lighting x command), tau_c={}'.format(args.tau_c))
    print('\nweather marginals :', marg['weather'], ' paper:', PAPER_COMPLETE['weather'])
    print('command marginals :', marg['command'], ' paper:', PAPER_COMPLETE['command'])
    n_no_can = sum(1 for a in attrs.values() if a['n_no_can'] > 0)
    print('scenes without CAN bus data (speed fallback used): {}'.format(n_no_can))
    alloc, exact = allocate(table, args.budget, args.gamma)
    print_table(exact, '\nunrounded n0*P_xy  (n0={}, gamma={})'.format(args.budget, args.gamma))
    print_table(alloc, '\nrounded allocation (largest remainder)')
    am = marginals(alloc)
    print('\nallocation marginals: weather', am['weather'], ' command', am['command'])
    print('paper gamma=0.5 row  : weather', PAPER_GAMMA05['weather'], ' command', PAPER_GAMMA05['command'])
    ok = marg == PAPER_COMPLETE
    print('\nMATCHES paper Table 4 "Complete" row: {}'.format(ok))
    return 0 if ok else 1


def cmd_select(args):
    _, attrs = build_attrs(args)
    table = count_table(attrs)
    alloc, exact = allocate(table, args.budget, args.gamma)
    selected = []
    for x in WEATHER_CELLS:
        for y in COMMAND_CELLS:
            cell = sorted((a for a in attrs.values() if a['weather'] == x and a['command'] == y),
                          key=lambda a: (a['mean_speed'], a['name']))
            for i in regular_interval_indices(len(cell), alloc[x][y]):
                a = cell[i]
                selected.append(dict(name=a['name'], token=a['token'],
                                     cell=x + '-' + y, weather=x, command=y,
                                     mean_speed=round(a['mean_speed'], 4), cmd_counts=a['cmd_counts'],
                                     n_frames=a['n_frames'], rank_in_cell=i, cell_size=len(cell),
                                     description=a['description']))
    sel_table = {x: {y: 0 for y in COMMAND_CELLS} for x in WEATHER_CELLS}
    for s in selected:
        sel_table[s['weather']][s['command']] += 1
    out = dict(
        method='ActiveAD Ego-Diversity (Sec. 4.1, Alg. 1)',
        params=dict(budget=args.budget, gamma=args.gamma, tau_c=args.tau_c, speed_source=args.speed_source,
                    rounding='largest-remainder per level', interval='floor((arange(k)+0.5)*n/k)',
                    train_pkl=os.path.abspath(args.train_pkl)),
        pool_counts_table=table, pool_marginals=marginals(table),
        alloc_unrounded=exact, alloc_rounded=alloc, selected_table=sel_table,
        selected_marginals=marginals(sel_table),
        paper_reference=dict(complete=PAPER_COMPLETE, gamma05=PAPER_GAMMA05),
        num_scenes=len(selected), num_frames=sum(s['n_frames'] for s in selected),
        scene_names=[s['name'] for s in selected], scene_tokens=[s['token'] for s in selected],
        scenes=selected)
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print_table(sel_table, 'selected scenes per cell')
    print('selected {} scenes / {} frames -> {}'.format(out['num_scenes'], out['num_frames'], args.out))
    print('marginals:', out['selected_marginals'], ' paper gamma=0.5:', PAPER_GAMMA05)
    return 0


def cmd_random(args):
    """Paper's 'Random' baseline: uniform sample of `budget` scenes (scene = selection unit)."""
    import random
    _, attrs = build_attrs(args)
    names = sorted(attrs.values(), key=lambda a: a['name'])
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(names, args.budget), key=lambda a: a['name'])
    selected = [dict(name=a['name'], token=a['token'], cell=a['weather'] + '-' + a['command'],
                     weather=a['weather'], command=a['command'], mean_speed=round(a['mean_speed'], 4),
                     cmd_counts=a['cmd_counts'], n_frames=a['n_frames'], description=a['description'])
                for a in picked]
    sel_table = {x: {y: 0 for y in COMMAND_CELLS} for x in WEATHER_CELLS}
    for a in selected:
        sel_table[a['weather']][a['command']] += 1
    out = dict(
        method='Random (uniform over train scenes, python random.Random(seed).sample)',
        params=dict(budget=args.budget, seed=args.seed, tau_c=args.tau_c,
                    train_pkl=os.path.abspath(args.train_pkl)),
        pool_counts_table=count_table(attrs), selected_table=sel_table,
        selected_marginals=marginals(sel_table),
        num_scenes=len(selected), num_frames=sum(a['n_frames'] for a in selected),
        scene_names=[a['name'] for a in selected], scene_tokens=[a['token'] for a in selected],
        scenes=selected)
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print_table(sel_table, 'random seed {}: selected scenes per cell'.format(args.seed))
    print('selected {} scenes / {} frames -> {}'.format(out['num_scenes'], out['num_frames'], args.out))
    return 0


def cmd_random_frames(args):
    """Frame-level random subset (scene boundaries ignored): `frac` of all train keyframes,
    plus the `context` preceding keyframes of each pick as context-only frames (no loss,
    not counted in the epoch) so the VAD temporal queue sees real 0.5 s history."""
    import random
    data = mmcv.load(args.train_pkl)
    infos = sorted(data['infos'], key=lambda e: e['timestamp'])   # same order the dataset uses
    n_target = int(round(args.frac * len(infos)))
    rng = random.Random(args.seed)
    target_pos = sorted(rng.sample(range(len(infos)), n_target))
    target_tok = {infos[i]['token'] for i in target_pos}
    context_tok = set()
    for i in target_pos:
        sc = infos[i]['scene_token']
        for k in range(1, args.context + 1):
            j = i - k
            if j >= 0 and infos[j]['scene_token'] == sc and infos[j]['token'] not in target_tok:
                context_tok.add(infos[j]['token'])
    scenes_covered = len({infos[i]['scene_token'] for i in target_pos})
    out = dict(method='Random over keyframes (scene boundaries ignored) + {} context-only predecessors'.format(args.context),
               params=dict(frac=args.frac, seed=args.seed, context=args.context, train_pkl=os.path.abspath(args.train_pkl)),
               num_targets=len(target_tok), num_context=len(context_tok), scenes_covered=scenes_covered,
               sample_tokens=sorted(target_tok), context_tokens=sorted(context_tok))
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('frame-random seed {}: {} target frames over {} scenes + {} context frames -> {}'.format(
        args.seed, len(target_tok), scenes_covered, len(context_tok), args.out))
    return 0


def cmd_filter(args):
    with open(args.scenes) as f:
        spec = json.load(f)
    data = mmcv.load(args.train_pkl)
    if isinstance(spec, dict) and 'sample_tokens' in spec:          # frame-level subset
        targets, context = set(spec['sample_tokens']), set(spec.get('context_tokens', []))
        infos = []
        for i in data['infos']:
            if i['token'] in targets:
                infos.append(dict(i, context_only=False))
            elif i['token'] in context:
                infos.append(dict(i, context_only=True))
        n_t = sum(1 for i in infos if not i['context_only'])
        out = dict(infos=infos, metadata=data['metadata'])
        mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
        mmcv.dump(out, args.out)
        print('kept {} target + {} context frames of {} -> {}'.format(n_t, len(infos) - n_t, len(data['infos']), args.out))
        return 0
    tokens = set(spec['scene_tokens'] if isinstance(spec, dict) else spec)
    infos = [i for i in data['infos'] if i['scene_token'] in tokens]
    kept = {i['scene_token'] for i in infos}
    missing = tokens - kept
    if missing:
        print('WARNING: {} scene tokens not found in pkl'.format(len(missing)))
    out = dict(infos=infos, metadata=data['metadata'])
    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    mmcv.dump(out, args.out)
    print('kept {} scenes / {} of {} frames -> {}'.format(len(kept), len(infos), len(data['infos']), args.out))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--train-pkl', required=True)
    common.add_argument('--nusc-dir', default='data/nuscenes/v1.0-trainval', help='dir containing scene.json')
    common.add_argument('--tau-c', type=int, default=4)
    common.add_argument('--gamma', type=float, default=0.5)
    common.add_argument('--budget', type=int, default=70, help='n0: number of scenes to select')
    common.add_argument('--speed-source', choices=['can_v0', 'pose'], default='can_v0')
    sub.add_parser('check', parents=[common]).set_defaults(func=cmd_check)
    ps = sub.add_parser('select', parents=[common])
    ps.add_argument('--out', default='tools/active/splits/ed10_gamma0.5.json')
    ps.set_defaults(func=cmd_select)
    pr = sub.add_parser('random', parents=[common])
    pr.add_argument('--seed', type=int, required=True)
    pr.add_argument('--out', required=True)
    pr.set_defaults(func=cmd_random)
    pq = sub.add_parser('random-frames')
    pq.add_argument('--train-pkl', required=True)
    pq.add_argument('--seed', type=int, required=True)
    pq.add_argument('--frac', type=float, default=0.1)
    pq.add_argument('--context', type=int, default=2, help='preceding keyframes kept as context-only')
    pq.add_argument('--out', required=True)
    pq.set_defaults(func=cmd_random_frames)
    pf = sub.add_parser('filter')
    pf.add_argument('--train-pkl', required=True)
    pf.add_argument('--scenes', required=True, help='split json (scene_tokens) or json list of tokens')
    pf.add_argument('--out', required=True)
    pf.set_defaults(func=cmd_filter)
    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
