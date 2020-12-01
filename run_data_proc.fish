#!/bin/fish

## Data processing - distilroberta
#python src/main.py --data_path ./data/ada_transformersum_redacted_splits/ \
#--model_name_or_path distilroberta-base \
#--model_type distilroberta \
#--no_use_token_type_ids \
#--use_logger tensorboard \
#--do_train --fast_dev_run



## Data processing - bert
python src/main.py --data_path ./data/ada_transformersum_redacted_splits_bert/ \
 --model_name_or_path ./ada_bert_pytorch/ --model_type bert --max_seq_length 512 \
 --use_logger tensorboard --do_train --fast_dev_run
