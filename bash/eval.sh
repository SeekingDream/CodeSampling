
for model_id in {0..2}; do
  for lora_data_id in {0..1}; do
    for partial_id in {0..4}; do
      for eval_data_id in {0..3}; do
        # Run the evaluation script with specified parameters
        python eval_pass_K.py \
          --model_id="$model_id" \
          --lora_data_id="$lora_data_id" \
          --partial_id="$partial_id" \
          --data_id="$eval_data_id"
      done
    done
  done
done


for model_id in {3..8}; do
  for eval_data_id in {0..3}; do
    # Run the evaluation script with specified parameters
    python eval_pass_K.py \
      --model_id="$model_id" \
      --lora_data_id=0 \
      --partial_id=0 \
      --data_id="$eval_data_id"
  done
done


