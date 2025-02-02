#!/bin/bash

# 设置参数
NUM_MODELS=8  # model_id 从 0 到 12，总共 13 个值
SAVE_DIR="./results/generated_code"

# 遍历 model_id 并运行 srun
for MODEL_ID in $(seq 0 $((NUM_MODELS)))
do
    echo "Running model_id=$MODEL_ID..."
    srun --gres=gpu:a6000:1 --time=2-0:0 \
        python generate_code.py --n=100 --temperature=0.8 \
        --model_id=$MODEL_ID --data_id=1 --save_dir="$SAVE_DIR" \
        --overwrite=True
done

echo "All jobs completed."
