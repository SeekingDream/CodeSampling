#!/bin/bash

# 设置参数
NUM_MODELS=8  # model_id 从 0 到 12，总共 13 个值
SAVE_DIR="./results/generated_code"

# 遍历 model_id 并运行 srun
for MODEL_ID in $(seq 0 $((NUM_MODELS)))
do
    echo "Evaluating model_id=$MODEL_ID..."
    python evaluate_sampling.py \
    --data_id=0 --n=100 \
    --sample_dir="./results/generated_code" \
    --model_id=2 \
    --temperature=0.8 \
    --eval_dir="./results/exe_res_dir" \
    --overwrite=False
done

echo "All jobs completed."
