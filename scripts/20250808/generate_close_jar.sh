SAVE_PATH=./data/20250808_close_jar/train/

for task in close_jar 
do
xvfb-run -a python3 tools/dataset_generator.py \
    --tasks=$task \
    --episodes_per_task=2 \
    --save_path=$SAVE_PATH \
    --image_size=256,256\
    --renderer=opengl \
    --processes=1 \
    --all_variations=True
done