#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/yolo/yolo26
exec /home/xinnan/anaconda3/envs/mamba-ssm/bin/python scripts/run_bac_rt_stage2.py
