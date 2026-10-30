#!/usr/bin/env bash
# Start the frame-level-random queue once the gamma=1 chain (chain_gamma1.sh -> vad_queue3) is finished.
cd /home/junspring/VAD
echo "[$(date '+%F %T')] chain4 waiting for chain_gamma1.sh / vad_queue3 ..."
while pgrep -f chain_gamma1.sh >/dev/null; do sleep 300; done
while docker ps --format '{{.Names}}' | grep -qx vad_queue3; do sleep 120; done
echo "[$(date '+%F %T')] starting vad_queue4 (frand10 seeds 0 1 2)"
NAME=vad_queue4 docker/run.sh tools/active/run_queue.sh frand10_s0:0 frand10_s1:1 frand10_s2:2
echo "[$(date '+%F %T')] chain4 done"
