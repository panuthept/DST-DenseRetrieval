python eval_msmarco.py --qrels_path ./test_data/marco_dev/qrels.dev.small.tsv --qpreds_path ./rank_bert/msmarco_rank.txt

echo 'DL-typo results'
echo 'The performance of original queries'
./trec_eval/trec_eval -l 2 -m ndcg_cut.10 -m map -m recip_rank ./test_data/dl-typo/qrels.txt ./rank_bert/dltypo_rank.txt.trec
echo 'The performance of misspelled queries'
./trec_eval/trec_eval -l 2 -m ndcg_cut.10 -m map -m recip_rank ./test_data/dl-typo/qrels.txt ./rank_bert/dltypo_typo_rank.txt.trec