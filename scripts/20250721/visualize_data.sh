SAVE_PATH=./data/20250721_robot_data/train/


for task in reach_single_moving_target_on_the_table_cpst \
    reach_single_moving_target_with_gravity \
    reach_single_bouncing_ball
do
python3 tools/dataset_visualizer.py visualize_episodes \
    --task=${task} \
    --root_dir=${SAVE_PATH}/${task}/all_variations/episodes \
    --save_dir=./vis_${task}/
done