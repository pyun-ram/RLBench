SAVE_PATH=./data/

for task in reach_three_moving_targets_on_the_table_nocpst reach_three_moving_targets_on_the_table_cpst
do
python3 tools/dataset_visualizer.py visualize_episodes \
    --task=${task} \
    --root_dir=data/${task}/all_variations/episodes \
    --save_dir=./vis_${task}/
done