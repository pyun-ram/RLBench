SAVE_PATH=./data/20250721_robot_data/train/

for task in reach_single_moving_target_on_the_table_cpst 
do
xvfb-run -a python3 tools/dataset_generator.py \
    --tasks=$task \
    --episodes_per_task=100 \
    --save_path=$SAVE_PATH \
    --image_size=256,256\
    --renderer=opengl \
    --processes=1 \
    --all_variations=True
done