"""Scene-level "cheap labels" used by ActiveAD's Ego-Diversity (Sec. 4.1).

Shared by ego_diversity_select.py (train split) and eval_breakdown.py (val split).

* weather / lighting: from the free-text ``description`` in nuScenes ``scene.json``.
  ``night`` -> Night else Day, ``rain`` -> Rainy else Sunny. There are no explicit
  day/sunny tags in nuScenes; they are the absence of the keyword. On the 700 train
  scenes this gives DS/DR/NS/NR = 491/125/71/13, exactly the paper's Table 4.
* driving command: VAD's per-keyframe ``gt_ego_fut_cmd`` (one-hot, see
  tools/data_converter/vad_nuscenes_converter.py: [1,0,0]=Turn Right,
  [0,1,0]=Turn Left, [0,0,1]=Go Straight), counted per scene and mapped to a
  clip category with the paper's rule (literal reading, tau_c=4):
  Overtake if #L >= tau_c and #R >= tau_c; Left if only #L > tau_c;
  Right if only #R > tau_c; else Straight -> S/R/L/O = 423/132/112/33 (Table 4).
* average speed: scene mean of the CAN-bus longitudinal speed ``v0``
  (``gt_ego_lcf_feat[7]``). For the few CAN-blacklisted scenes the converter
  writes zeros into ``can_bus`` and a half-scaled ``v0``; there we fall back to
  the pose-derived |(vx, vy)| in ``gt_ego_lcf_feat[0:2]`` (m/s).
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

CMD_IDX = {0: 'R', 1: 'L', 2: 'S'}
WEATHER_CELLS = ['DS', 'DR', 'NS', 'NR']
COMMAND_CELLS = ['S', 'R', 'L', 'O']
WEATHER_NAMES = {'DS': 'Day-Sunny', 'DR': 'Day-Rainy', 'NS': 'Night-Sunny', 'NR': 'Night-Rainy'}
COMMAND_NAMES = {'S': 'Go Straight', 'R': 'Turn Right', 'L': 'Turn Left', 'O': 'Overtake'}


def weather_lighting(description):
    d = description.lower()
    return ('N' if 'night' in d else 'D') + ('R' if 'rain' in d else 'S')


def clip_command(cmd_counts, tau_c=4):
    L, R = cmd_counts.get('L', 0), cmd_counts.get('R', 0)
    if L >= tau_c and R >= tau_c:
        return 'O'
    if L > tau_c:
        return 'L'
    if R > tau_c:
        return 'R'
    return 'S'


def load_scenes(nusc_version_dir):
    """scene.json -> {scene_token: dict(name, description, nbr_samples, log_token)}."""
    with open(os.path.join(nusc_version_dir, 'scene.json')) as f:
        scenes = json.load(f)
    return {s['token']: dict(name=s['name'], description=s['description'],
                             nbr_samples=s['nbr_samples'], log_token=s['log_token'])
            for s in scenes}


def frame_speed(info, speed_source='can_v0'):
    lcf = np.asarray(info['gt_ego_lcf_feat'], dtype=np.float64)
    has_can = np.any(np.asarray(info['can_bus']) != 0)
    if speed_source == 'can_v0' and has_can:
        return float(lcf[7])
    return float(np.hypot(lcf[0], lcf[1]))


def per_scene_stats(infos, tau_c=4, speed_source='can_v0'):
    """Aggregate VAD info dicts -> {scene_token: dict(cmd_counts, clip_cmd, mean_speed, n_frames, n_no_can)}."""
    cmd_counts = defaultdict(Counter)
    speeds = defaultdict(list)
    no_can = Counter()
    for info in infos:
        tok = info['scene_token']
        cmd_counts[tok][CMD_IDX[int(np.argmax(info['gt_ego_fut_cmd']))]] += 1
        speeds[tok].append(frame_speed(info, speed_source))
        if not np.any(np.asarray(info['can_bus']) != 0):
            no_can[tok] += 1
    stats = {}
    for tok, cnt in cmd_counts.items():
        stats[tok] = dict(
            cmd_counts={k: int(cnt.get(k, 0)) for k in ('S', 'R', 'L')},
            clip_cmd=clip_command(cnt, tau_c),
            mean_speed=float(np.mean(speeds[tok])),
            n_frames=len(speeds[tok]),
            n_no_can=int(no_can.get(tok, 0)))
    return stats


def scene_attributes(infos, scenes, tau_c=4, speed_source='can_v0'):
    """Join per-scene stats with scene.json -> {scene_token: dict(name, weather, command, ...)}."""
    stats = per_scene_stats(infos, tau_c, speed_source)
    out = {}
    for tok, st in stats.items():
        sc = scenes[tok]
        out[tok] = dict(token=tok, name=sc['name'], description=sc['description'],
                        weather=weather_lighting(sc['description']),
                        command=st['clip_cmd'], **st)
    return out
