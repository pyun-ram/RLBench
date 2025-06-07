SAVE_PATH=./data/

for task in reach_single_moving_target_on_the_table_nocpst reach_single_moving_target_on_the_table_cpst
do
python3 tools/dataset_generator.py \
    --tasks=$task \
    --save_path=$SAVE_PATH \
    --image_size=256,256\
    --renderer=opengl \
    --processes=1 \
    --all_variations=True
done