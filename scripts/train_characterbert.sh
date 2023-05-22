#!/bin/sh

export WANDB_DISABLED=true

python -m tevatron_dst.driver.train \
--model_name_or_path bert-base-uncased \
--character_bert_path ./general_character_bert \
--output_dir model_msmarco_characterbert \
--passage_field_separator [SEP] \
--save_steps 40000 \
--dataset_name Tevatron/msmarco-passage \
--fp16 \
--per_device_train_batch_size 16 \
--learning_rate 1e-5 \
--max_steps 150000 \
--dataloader_num_workers 10 \
--cache_dir ./cache \
--logging_steps 150 \
--character_query_encoder True \
--query_expansion_size 40 \
--beta 0.5 \
--gamma 0.5 \
--sigma 0.2