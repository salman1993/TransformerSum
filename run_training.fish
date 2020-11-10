#!/bin/fish

## Fast run
python src/main.py --data_path ./data/test_one_sent/ --do_train --max_steps 3 --gpus 0 --use_logger tensorboard

# Training
python src/main.py \
--model_name_or_path distilroberta-base \
--model_type distilroberta \
--no_use_token_type_ids \
--use_logger tensorboard \
--use_custom_checkpoint_callback \
--max_epochs 3 \
--accumulate_grad_batches 2 \
--warmup_steps 1400 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--do_train \
--batch_size 16 \
--classifier simple_linear \
--data_path ./data/ada_transformersum_dataset/ \
--gpus 0