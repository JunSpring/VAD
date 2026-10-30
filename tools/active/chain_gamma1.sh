#!/usr/bin/env bash
# Start the gamma=1 queue once vad_queue (seeds 1/2) and vad_queue2 (rand s0) are both gone.
cd /home/junspring/VAD
echo "[$(date '+%F %T')] chain3 waiting for vad_queue / vad_queue2 ..."
# wait until vad_queue has exited
while docker ps --format '{{.Names}}' | grep -qx vad_queue; do sleep 120; done
echo "[$(date '+%F %T')] vad_queue gone; waiting for vad_queue2 to start and finish"
sleep 300   # give chain2 time to launch vad_queue2
while docker ps --format '{{.Names}}' | grep -qx vad_queue2; do sleep 120; done
echo "[$(date '+%F %T')] starting vad_queue3 (ed10_g1 seeds 0 1 2)"
NAME=vad_queue3 docker/run.sh tools/active/run_queue.sh ed10_g1:0 ed10_g1:1 ed10_g1:2
echo "[$(date '+%F %T')] chain3 done"
