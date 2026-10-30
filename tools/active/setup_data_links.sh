#!/usr/bin/env bash
# Build the data/ layout VAD expects (docs/prepare_dataset.md) out of symlinks
# to the nuScenes copy on the JunBoom SSD. Idempotent; run on the HOST from the
# repo root. The container mounts the SSD at the same absolute path, so the
# links resolve there too.
#
#   data/can_bus                    -> $NUSC/can_bus            (download can_bus.zip first)
#   data/nuscenes/v1.0-trainval     -> $NUSC/Trainval/v1.0-trainval_meta/v1.0-trainval
#   data/nuscenes/{samples,sweeps}  -> $NUSC/Trainval/{samples,sweeps}
#   data/nuscenes/maps/*.png        -> $NUSC/Trainval/v1.0-trainval_meta/maps/*.png
#   data/nuscenes/maps/expansion    -> $NUSC/maps_expansion    (map expansion v1.3)
#   data/nuscenes/*.pkl             -> $NUSC/vad_infos/*.pkl   (converter output)
#   data/nuscenes_mini/...          same, for the v1.0-mini smoke test
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NUSC=${NUSC:-/media/junspring/JunBoom/dataset/nuScenes}
cd "$REPO"

# data/nuscenes must be a real directory: the first evaluation writes
# nuscenes_map_anns_val.json into data_root.
mkdir -p data/nuscenes/maps data/nuscenes_mini/maps ckpt/torch/hub/checkpoints

ln -sfn "$NUSC/can_bus" data/can_bus

# ---- full trainval -------------------------------------------------------
ln -sfn "$NUSC/Trainval/v1.0-trainval_meta/v1.0-trainval" data/nuscenes/v1.0-trainval
ln -sfn "$NUSC/Trainval/samples" data/nuscenes/samples
ln -sfn "$NUSC/Trainval/sweeps"  data/nuscenes/sweeps
for p in "$NUSC"/Trainval/v1.0-trainval_meta/maps/*.png; do ln -sfn "$p" data/nuscenes/maps/; done
ln -sfn "$NUSC/maps_expansion" data/nuscenes/maps/expansion
for s in train val train_ed10; do
  ln -sfn "$NUSC/vad_infos/vad_nuscenes_infos_temporal_$s.pkl" data/nuscenes/vad_nuscenes_infos_temporal_$s.pkl
done
# any other subset pkl produced by `ego_diversity_select.py filter` (e.g. train_rand10_s1)
for p in "$NUSC"/vad_infos/vad_nuscenes_infos_temporal_train_*.pkl; do
  [ -e "$p" ] && ln -sfn "$p" "data/nuscenes/$(basename "$p")"
done

# ---- v1.0-mini (smoke test) ---------------------------------------------
ln -sfn "$NUSC/v1.0-mini/v1.0-mini" data/nuscenes_mini/v1.0-mini
ln -sfn "$NUSC/v1.0-mini/samples"   data/nuscenes_mini/samples
ln -sfn "$NUSC/v1.0-mini/sweeps"    data/nuscenes_mini/sweeps
for p in "$NUSC"/v1.0-mini/maps/*.png; do ln -sfn "$p" data/nuscenes_mini/maps/; done
ln -sfn "$NUSC/maps_expansion" data/nuscenes_mini/maps/expansion
for s in train val; do
  ln -sfn "$NUSC/vad_infos_mini/vad_nuscenes_infos_temporal_$s.pkl" data/nuscenes_mini/vad_nuscenes_infos_temporal_$s.pkl
done

# torchvision://resnet50 -> $TORCH_HOME/hub/checkpoints (TORCH_HOME=ckpt/torch in the image)
if [ -f ckpt/resnet50-19c8e357.pth ]; then
  cp -n ckpt/resnet50-19c8e357.pth ckpt/torch/hub/checkpoints/ || true
fi

echo "--- data layout ---"
ls -l data data/nuscenes data/nuscenes/maps | sed 's/^/  /'
echo "--- required inputs ---"
for f in data/nuscenes/v1.0-trainval/scene.json data/nuscenes/maps/expansion/boston-seaport.json \
         data/can_bus/scene-0001_meta.json ckpt/torch/hub/checkpoints/resnet50-19c8e357.pth; do
  if [ -e "$f" ]; then echo "  OK       $f"; else echo "  MISSING  $f"; fi
done
