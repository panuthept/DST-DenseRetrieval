#!/bin/sh

export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=$2

NAME="bert"
LOAD_MODEL_DIR="model_msmarco_${NAME}"
SAVE_EMBEDDING_DIR="msmarco_${NAME}_embs" 
SAVE_RANK_DIR="rank_${NAME}"
echo "LOAD_MODEL_DIR: ${LOAD_MODEL_DIR}"
echo "SAVE_EMBEDDING_DIR: ${SAVE_EMBEDDING_DIR}"
echo "SAVE_RANK_DIR: ${SAVE_RANK_DIR}"

mkdir $SAVE_EMBEDDING_DIR
mkdir $SAVE_RANK_DIR

# encode query dl-typo
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $LOAD_MODEL_DIR/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path test_data/dl-typo/query.typo.tsv \
  --encoded_save_path $SAVE_EMBEDDING_DIR/query_dltypo_typo_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry 

python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $LOAD_MODEL_DIR/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path test_data/dl-typo/query.tsv \
  --encoded_save_path $SAVE_EMBEDDING_DIR/query_dltypo_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry 

# encode query msmarco dev
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $LOAD_MODEL_DIR/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path test_data/marco_dev/queries.dev.small.tsv \
  --encoded_save_path ${SAVE_EMBEDDING_DIR}/query_msmarco_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry 

# encode query msmarco-typo dev
for s in $(seq 1 10)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $LOAD_MODEL_DIR/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path test_data/marco_dev/queries.dev.small.typo${s}.tsv \
  --encoded_save_path ${SAVE_EMBEDDING_DIR}/query_msmarco_typo${s}_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry 
done

# encode corpus
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $LOAD_MODEL_DIR/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --p_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${SAVE_EMBEDDING_DIR}/corpus_emb.${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s} \
  --cache_dir cache \
  --passage_field_separator [SEP]
done

# retrieve query dl-typo
python -m tevatron.faiss_retriever \
  --query_reps ${SAVE_EMBEDDING_DIR}/query_dltypo_typo_emb.pkl \
  --passage_reps ${SAVE_EMBEDDING_DIR}/'corpus_emb.*.pkl' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${SAVE_RANK_DIR}/dltypo_typo_rank.txt

python -m tevatron.faiss_retriever \
  --query_reps ${SAVE_EMBEDDING_DIR}/query_dltypo_emb.pkl \
  --passage_reps ${SAVE_EMBEDDING_DIR}/'corpus_emb.*.pkl' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${SAVE_RANK_DIR}/dltypo_rank.txt

# retrieve query msmarco dev
python -m tevatron.faiss_retriever \
  --query_reps ${SAVE_EMBEDDING_DIR}/query_msmarco_emb.pkl \
  --passage_reps ${SAVE_EMBEDDING_DIR}/'corpus_emb.*.pkl' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${SAVE_RANK_DIR}/msmarco_rank.txt

# retrieve query msmarco-typo dev
for s in $(seq 1 10)
do
python -m tevatron.faiss_retriever \
  --query_reps ${SAVE_EMBEDDING_DIR}/query_msmarco_typo${s}_emb.pkl \
  --passage_reps ${SAVE_EMBEDDING_DIR}/'corpus_emb.*.pkl' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${SAVE_RANK_DIR}/msmarco_typo${s}_rank.txt
done

python -m tevatron.utils.format.convert_result_to_trec \
  --input ${SAVE_RANK_DIR}/dltypo_typo_rank.txt \
  --output ${SAVE_RANK_DIR}/dltypo_typo_rank.txt.trec

python -m tevatron.utils.format.convert_result_to_trec \
  --input ${SAVE_RANK_DIR}/dltypo_rank.txt \
  --output ${SAVE_RANK_DIR}/dltypo_rank.txt.trec