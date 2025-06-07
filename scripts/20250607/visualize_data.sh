SAVE_PATH=./data/

for task in reach_single_moving_target_on_the_table_nocpst reach_single_moving_target_on_the_table_cpst
do
python3 tools/dataset_visualizer.py visualize_episodes \
    --root_dir=data/${task}/all_variations/episodes \
    --save_dir=./vis_${task}/
done