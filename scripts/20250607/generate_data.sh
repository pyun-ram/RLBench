SAVE_PATH=./data/

for task in reach_single_moving_target_on_the_table_cpst \
            reach_two_moving_targets_on_the_table \
            reach_single_moving_target_with_gravity \
            reach_two_moving_targets_with_gravity \
            reach_single_bouncing_ball
do
python3 tools/dataset_generator.py \
    --tasks=$task \
    --episodes_per_task=1 \
    --save_path=$SAVE_PATH \
    --image_size=256,256\
    --renderer=opengl \
    --processes=1 \
    --all_variations=True
done