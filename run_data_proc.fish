#!/bin/fish

# Data processing - bert
python src/main.py --data_path ./data/cnn_dm_extractive_compressed_5000/ \
--model_name_or_path ./ada_bert_pytorch/ --model_type bert --max_seq_length 512 \
--use_logger tensorboard --do_train --fast_dev_run --gpus 0


##  Data processing - distilroberta
# python src/main.py --data_path ./data/ada_transformersum_redacted_splits/ \
# --model_name_or_path distilroberta-base --model_type distilroberta --no_use_token_type_ids \
# --use_logger tensorboard --do_train --fast_dev_run --gpus 0