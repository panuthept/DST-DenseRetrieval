import os
import numpy as np

from argparse import ArgumentParser
from ranx import Qrels, Run, evaluate


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--qrels_path', default="./data/marco_dev/qrels.dev.small.tsv")
    parser.add_argument('--qpreds_path', required=True)
    args = parser.parse_args()

    print("MS MARCO results")
    print("The performance of original queries")
    qrels_data = open(args.qrels_path, 'rt').readlines()
    qpreds_data = open(args.qpreds_path, 'rt').readlines()

    qrels_data = [line.replace('\n', '').split('\t') for line in qrels_data]
    qpreds_data = [line.replace('\n', '').split('\t') for line in qpreds_data]

    qrels_dict = {}
    for i in qrels_data:
        if i[0] in qrels_dict.keys():
            qrels_dict[i[0]].update({i[2]: i[3]})
        else:
            qrels_dict[i[0]] = {i[2]: i[3]}
            
    qpreds_dict = {}
    for i in qpreds_data:
        if i[0] in qpreds_dict.keys():
            qpreds_dict[i[0]].update({i[1]: i[2]})
        else:
            qpreds_dict[i[0]] = {i[1]: i[2]}
            
    qrels = Qrels(qrels_dict)
    run = Run(qpreds_dict)
    results = evaluate(qrels, run, ["mrr@10", "recall@1000"])
    print(f"mrr@10: {results['mrr@10']:.03f}, recall@1000: {results['recall@1000']:.03f}")

    print("The performance of misspelled queries")
    avg_results = {"mrr@10": [], "recall@1000": []}
    for j in range(1, 11):
        qpreds_data = open(args.qpreds_path.replace("rank.txt", f"typo{j}_rank.txt"), 'rt').readlines()

        qpreds_data = [line.replace('\n', '').split('\t') for line in qpreds_data]
                
        qpreds_dict = {}
        for i in qpreds_data:
            if i[0] in qpreds_dict.keys():
                qpreds_dict[i[0]].update({i[1]: i[2]})
            else:
                qpreds_dict[i[0]] = {i[1]: i[2]}
                
        run = Run(qpreds_dict)
        results = evaluate(qrels, run, ["mrr@10", "recall@1000"])
        print(f"(Typo{j}) mrr@10: {results['mrr@10']:.03f}, recall@1000: {results['recall@1000']:.03f}")
        avg_results["mrr@10"].append(results["mrr@10"])
        avg_results["recall@1000"].append(results["recall@1000"])

    avg_results = {k: np.mean(v) for k, v in avg_results.items()}
    print(f"(Averaged) mrr@10: {avg_results['mrr@10']:.03f}, recall@1000: {avg_results['recall@1000']:.03f}")