#!/bin/fish

## distilroberta
python src/main.py \
--model_name_or_path distilroberta-base \
--model_type distilroberta \
--no_use_token_type_ids \
--data_path ./data/roberta-base-cnn-dm-ada/ \
--pooling_mode sent_rep_tokens \
--use_logger tensorboard \
--use_custom_checkpoint_callback \
--max_epochs 15 \
--accumulate_grad_batches 2 \
--warmup_steps 1400 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--do_train --do_test \
--batch_size 16 \
--classifier simple_linear
