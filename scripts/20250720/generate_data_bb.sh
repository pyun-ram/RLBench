SAVE_PATH=./data/20250719_robot_data/train/

for task in reach_single_bouncing_ball
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