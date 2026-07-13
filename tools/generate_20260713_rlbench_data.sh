#!/usr/bin/env bash
# Run with:
# conda run -n rlbench bash tools/generate_20260713_rlbench_data.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data/20260713_rlbench_data_fix"
TASKS="beat_the_buzz"

cd "${REPO_ROOT}"
python tools/dataset_generator.py \
    --save_path="${DATA_ROOT}" \
    --tasks="${TASKS// /,}" \
    --episodes_per_task=1 \
    --variations=1 \
    --processes=1

for task in ${TASKS}; do
    for episode in 0; do
        python tools/visualize_waypoint_demo.py \
            --episode_dir="${DATA_ROOT}/${task}/variation0/episodes/episode${episode}" \
            --point_stride=8 \
            --frame_stride=4
    done
done
