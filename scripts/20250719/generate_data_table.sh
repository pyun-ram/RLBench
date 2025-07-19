SAVE_PATH=./data/20250719_robot_data/train/

for task in reach_single_moving_target_on_the_table_cpst \
    reach_single_moving_target_with_gravity \
    reach_single_bouncing_ball
do
xvfb-run -a python3 tools/dataset_generator.py \
    --tasks=$task \
    --episodes_per_task=70 \
    --save_path=$SAVE_PATH \
    --image_size=256,256\
    --renderer=opengl \
    --processes=1 \
    --all_variations=True
done