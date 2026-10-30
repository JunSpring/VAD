#!/usr/bin/env bash
# Sequential train -> eval -> breakdown queue for the ActiveAD 10% experiments (run INSIDE the container).
#   tools/active/run_queue.sh ed10:1 rand10_s1:1 ed10:2 rand10_s2:2
# Each item is <split>:<seed>; <split> names data/nuscenes/vad_nuscenes_infos_temporal_train_<split>.pkl
# and the run dir work_dirs/vad_tiny_<split>_seed<seed>. A failing step is logged and the queue moves on.
set -uo pipefail
CFG=projects/configs/VAD/VAD_tiny_e2e_activead_ed10.py
PORT=${PORT:-29521}
for item in "$@"; do
  split=${item%%:*}; seed=${item##*:}
  run=work_dirs/vad_tiny_${split}_seed${seed}
  ann=data/nuscenes/vad_nuscenes_infos_temporal_train_${split}.pkl
  mkdir -p "$run"
  echo "=== [$(date '+%F %T')] START $item -> $run (ann=$ann)" | tee -a "$run/queue.log"
  if [ ! -f "$run/epoch_20.pth" ]; then
    python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT tools/train.py "$CFG" \
      --launcher pytorch --deterministic --seed "$seed" --no-validate --work-dir "$run" \
      --cfg-options data.train.ann_file="$ann" > "$run/train.log" 2>&1 \
      || { echo "!!! train FAILED for $item (see $run/train.log)" | tee -a "$run/queue.log"; continue; }
  fi
  if [ ! -f "$run/eval_ep20/breakdown.json" ]; then
    CUDA_VISIBLE_DEVICES=0 python tools/test.py "$CFG" "$run/epoch_20.pth" --launcher none --eval bbox \
      --json_dir "$run/eval_ep20" > "$run/eval.log" 2>&1 \
      || { echo "!!! eval FAILED for $item (see $run/eval.log)" | tee -a "$run/queue.log"; continue; }
    python tools/active/eval_breakdown.py --metric-json "$run/eval_ep20/metric_record.json" \
      --val-pkl data/nuscenes/vad_nuscenes_infos_temporal_val.pkl --nusc-dir data/nuscenes/v1.0-trainval \
      >> "$run/eval.log" 2>&1
  fi
  echo "=== [$(date '+%F %T')] DONE $item: $(grep -h -E '^All ' "$run/eval.log" | tail -1)" | tee -a "$run/queue.log"
done
echo "=== [$(date '+%F %T')] QUEUE FINISHED"
