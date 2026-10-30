#!/usr/bin/env bash
# Start the VAD container. Repo and the nuScenes SSD are mounted at the SAME
# absolute paths as on the host so the symlinks created by
# tools/active/setup_data_links.sh resolve inside the container.
#
#   docker/run.sh                 # interactive bash
#   docker/run.sh python ...      # run a command
#   NAME=vad_train docker/run.sh  # named container (for `docker logs -f`)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-vad:cu118-torch2.0.1}
NUSC=${NUSC:-/media/junspring/JunBoom/dataset/nuScenes}
NAME=${NAME:-}

# Docker creates a missing bind-mount source as a root-owned directory, which
# then blocks the user (and rsync) from writing there. Refuse instead.
if [ ! -d "$NUSC" ]; then
  echo "run.sh: dataset dir $NUSC does not exist; create it (as $(id -un)) or set NUSC=..." >&2
  exit 1
fi

ARGS=(--gpus all --rm --shm-size=32g
  -e LOCAL_USER_ID="$(id -u)" -e LOCAL_GROUP_ID="$(id -g)"
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  --hostname "$(hostname)"
  -v "$REPO:$REPO"
  -v "$NUSC:$NUSC"
  -w "$REPO")
[ -n "$NAME" ] && ARGS+=(--name "$NAME")
# Publish a port only on request (TB_PORT=6006 docker/run.sh tensorboard ...);
# a fixed -p would make a second concurrent container fail to start.
[ -n "${TB_PORT:-}" ] && ARGS+=(-p "$TB_PORT:$TB_PORT")
# interactive tty only when attached to a terminal
if [ -t 0 ]; then ARGS+=(-it); else ARGS+=(-i); fi

docker run "${ARGS[@]}" "$IMAGE" "${@:-bash}"
