#!/bin/fish

## bert
python src/main.py \
 --model_name_or_path ./ada_bert_pytorch/ \
 --model_type bert \
 --max_seq_length 512 \
--data_path ./data/ada-bert-base-cnn-dm-ada/ \
--pooling_mode sent_rep_tokens \
--use_logger tensorboard \
--use_custom_checkpoint_callback \
--max_epochs 20 \
--accumulate_grad_batches 2 \
--warmup_steps 1400 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--do_train --do_test \
--batch_size 8 \
--num_frozen_steps 10000 \
--classifier transformer \
--classifier_transformer_num_layers 2