#!/bin/fish

python src/main.py \
--data_path ./data/roberta-base-cnn-dm-ada/ \
--use_logger tensorboard \
--do_test \
--batch_size 16 \
--load_from_checkpoint test_ckpts/trained_models-v1.ckpt
