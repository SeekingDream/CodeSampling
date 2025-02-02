

N=20
model_id=1
data_id=2
gpu_id=1

# Iterate over different lora_data_id and partial_id values
for lora_data_id in 0 1; do
  for partial_id in {0..4}; do
    # Set CUDA_VISIBLE_DEVICES and run the python script
    CUDA_VISIBLE_DEVICES=${gpu_id} python generate_code.py \
      --n=$N \
      --model_id=$model_id \
      --partial_id="$partial_id" \
      --lora_data_id=$lora_data_id \
      --data_id=$data_id
  done
done


model_id=4
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