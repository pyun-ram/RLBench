SAVE_PATH=./data/

for task in reach_single_moving_target_with_gravity
do
python3 tools/dataset_visualizer.py visualize_episodes \
    --task=${task} \
    --root_dir=data/${task}/all_variations/episodes \
    --save_dir=./vis_${task}/
done