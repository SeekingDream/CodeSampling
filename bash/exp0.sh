# CUDA_VISIBLE_DEVICES="${model_id},$(expr 7 - $model_id)" python overfit.py --model_id=$model_id --data_id=$data_id

N=20
model_id=0
data_id=2
gpu_id=0

for lora_data_id in 0 1; do
  for partial_id in {0..4}; do
    # Set CUDA_VISIBLE_DEVICES and run the python script
    CUDA_VISIBLE_DEVICES=${gpu_id} python generate_code.py \
      --n=$N \
      --model_id=$model_id \
      --partial_id=$partial_id \
      --lora_data_id=$lora_data_id \
      --data_id=$data_id
  done
done


model_id=3
partial_id=0
lora_data_id=0
gpu_id=$((model_id - 3))
for data_id in 2 3; do
  CUDA_VISIBLE_DEVICES=${gpu_id} python generate_code.py \
      --n=$N \
      --model_id=$model_id \
      --partial_id=$partial_id \
      --lora_data_id=$lora_data_id \
      --data_id=$data_id
done



