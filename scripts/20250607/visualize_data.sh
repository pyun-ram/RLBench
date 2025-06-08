SAVE_PATH=./data/

for task in  reach_single_moving_target_on_the_table_cpst \
            reach_two_moving_targets_on_the_table \
            reach_single_moving_target_with_gravity \
            reach_two_moving_targets_with_gravity \
            reach_single_bouncing_ball
do
python3 tools/dataset_visualizer.py visualize_episodes \
    --task=${task} \
    --root_dir=data/${task}/all_variations/episodes \
    --save_dir=./vis_${task}/
done