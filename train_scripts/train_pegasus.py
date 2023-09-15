# https://github.com/renmada/t5-pegasus-pytorch
import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from simplet5 import SimpleT5

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from bert_score import score as bertscore
import jieba
from transformers import MT5ForConditionalGeneration, BertTokenizer
from collections import defaultdict


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

def hit_at_k(y_true, y_pred, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    Returns:
        np.ndarray: hit score.
    """
    ground_truth = y_true
    predictions = y_pred[:k]   
    correct = list(set(ground_truth).intersection(set(predictions)))
    return len(correct)/len(ground_truth)

def MRR(y_true, y_pred):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = [[gold == pred for gold in y_true for pred in y_pred]]
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def create_dataset(df):
    domains = df['domain']
    src = df["source_text"]
    inputs = [f"预测上位词: <{d}> {s}" for d, s in zip(domains, src)]
    return inputs


def create_test_dataset(df_test):
    test_domains = df_test['domain']
    test_src = df_test["source_text"]
    test_tgt = df_test["target_text"]
    test_inputs = [f"预测上位词: <{d}> {s}" for d, s in zip(test_domains, test_src)]

    data_dict = defaultdict(list)
    for input, output in zip(test_inputs, test_tgt):
        data_dict[input].append(output)
    return list(data_dict.keys()), list(data_dict.values())


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--domain_specific', action="store_true")
parser.add_argument('--do_train', action="store_true")
parser.add_argument('--do_eval', action="store_true")
parser.add_argument('--checkpoint', required=False)
args = parser.parse_args()

if __name__ == '__main__':
    # load dataset
    df_train = pd.read_csv(os.path.join(args.data_dir, "train_hH.csv"))
    df_valid = pd.read_csv(os.path.join(args.data_dir, "valid_hH.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test_hH.csv"))

    # add end of sentence token at the end of input and output texts
    df_train["target_text"] = df_train["target_text"].apply(lambda x: str(x))
    df_valid["target_text"] = df_valid["target_text"].apply(lambda x: str(x))
    df_test["target_text"] = df_test["target_text"].apply(lambda x: str(x))

    output_dir = os.path.join(args.data_dir, "chinese_t5output")
    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_dir)


    # load model
    model = SimpleT5()
    # load (supports t5, mt5, byT5 models)
    model.load_model("zhuiyi-pegasus","zhuiyi_t5_pegasus")
    # model.add_tokens(["<end>"])
    # add special tokens into tokenizer
    model.add_tokens(["<A>", "<B>", "<C>","<D>","<E>","<F>","<G>","<H>","<Y>", "<end>"])

    df_train["source_text"] = create_dataset(df_train)
    df_valid["source_text"] = create_dataset(df_valid)

    torch.cuda.empty_cache()

    # load tokenizer
    tokenizer = T5PegasusTokenizer.from_pretrained('./zhuiyi_t5_pegasus')

    if args.do_train:
        # train
        model.train(train_df=df_train, # pandas dataframe with 2 columns: source_text & target_text
                    eval_df=df_valid, # pandas dataframe with 2 columns: source_text & target_text
                    source_max_token_len = 128, 
                    target_max_token_len = 128,
                    batch_size = 8,
                    max_epochs = 3,
                    use_gpu = True,
                    outputdir = output_dir,
                    early_stopping_patience_epochs = 0, # 0 to disable early stopping feature
                    precision = 32
                    )
    if args.do_eval:
        if args.checkpoint:
            model.load_model(os.path.join(output_dir,args.checkpoint), use_gpu=True)
        else:    
            model.load_model(output_dir, use_gpu=True)

    # predict
    predictions = []
    test_inputs, actuals = create_test_dataset(df_test)
    for input in tqdm(test_inputs):
        predictions.append(model.predict(input, num_return_sequences = 10, num_beams = 15, lang="zh"))

    h1_scores = [hit_at_k(a, p, k=1) for p,a in zip(predictions, actuals)]
    h3_scores = [hit_at_k(a, p, k=3) for p,a in zip(predictions, actuals)]
    h10_scores = [hit_at_k(a, p, k=10) for p,a in zip(predictions, actuals)]
    mrrs = [MRR(a, p) for p,a in zip(predictions, actuals)]
    # bert_scores = [float(sum([bertscore(p, [aa]*len(p), lang="en", verbose=False)[2].mean() for aa in a])/len(p)) for p,a in zip(predictions, actuals)]   # comment for the reason it is too slow
 
    final_df = pd.DataFrame({"source_text": test_inputs, "Generated Text": predictions, "Actual Text": actuals, "H@1": h1_scores, "H@3": h3_scores, "H@10": h10_scores, "MRR": mrrs})#, "BertScore": bert_scores})
    final_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    for col in ["H@1", "H@3", "H@10", "MRR"]:#, "BertScore"]:
        print(
            f"""{col}: {final_df[col].mean()}\n"""
        )
