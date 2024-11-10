# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj
import utils
from dataloader import CustomDataLoader, tokenizer_init
from dataloader_utils import Char_Token_SpanConverter

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="NYT", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        # triple: [subj_tok_span, obj_tok_span, rela]
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output


def span2str_cmim(triples, text, id2rel, span_converter: Char_Token_SpanConverter):
    output = []
    for triple in triples:
        # triple: (('H', 0, 5), ('T', 12, 17), 0)
        # print(triple)
        rel = id2rel[str(triple[-1])]
        subj_tok_span = (triple[0][1], triple[0][2])
        obj_tok_span = (triple[1][1], triple[1][2])
        subj_char_span = span_converter.get_char_span(text, subj_tok_span)
        obj_char_span = span_converter.get_char_span(text, obj_tok_span)
        subj = text[subj_char_span[0]:subj_char_span[1]]
        obj = text[obj_char_span[0]:obj_char_span[1]]
        # info = f"{subj}[sep]{rel}[sep]{obj}[sep]{str(subj_char_span)}[sep]{str(obj_char_span)}"
        info = f"{subj}[sep]{rel}[sep]{obj}"
        output.append(info)
    return output


def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode

    with open(params.data_dir / f'rel2id.json', 'r', encoding='utf-8') as f_re:
        id2rel = json.load(f_re)[0]

    # jyz chg
    tokenizer = tokenizer_init(os.path.join(params.bert_model_dir, 'vocab.txt'))
    span_converter = Char_Token_SpanConverter(
        tokenizer, add_special_tokens=False, has_return_offsets_mapping=False)

    # jyz chg
    label_datas = None
    if mark.lower() == 'val':
        with open(params.data_dir / f'{mark.lower()}_triples.json', "r", encoding='utf-8') as f:
            label_datas = json.load(f)
    elif mark.lower() == 'test':
        with open(params.data_dir / f'{mark.lower()}_triples.json', "r", encoding='utf-8') as f:
            label_datas = json.load(f)

    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []
    my_preds = []   # jyz chg
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            # jyz chg. 将生成的位置直接从数据集中抽取文本
            # print(f"input_tokens[idx] = {input_tokens[idx]}")
            my_pred_triples = span2str_cmim(
                pre_triples, label_datas[0]['text'], id2rel, span_converter)
            my_preds.append({
                'text': label_datas[0]['text'],
                'triples_pred_list': list(set(my_pred_triples)),
            })
            del label_datas[0]

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))

            # counter
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))
    metrics = get_metrics(correct_num, predict_num, gold_num)
    # logging loss, f1 and report
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics:\n".format(mark) + metrics_str)
    # return metrics, predictions, ground_truths
    return metrics, predictions, ground_truths, my_preds   # jyz chg


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'corres_threshold': args.mat_threshold,
        'rel_threshold': args.rel_pre_threshold,
        'ensure_corres': args.ensure_match,
        'ensure_rel': args.ensure_relpre,
        'emb_fusion': args.emb_fusion
    }

    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")
    _, predictions, ground_truths = evaluate(model, loader, params, ex_params, mark=mode)
    with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths
            }
        )
        df.to_csv(params.ex_dir / f'{mode}_result.csv')
    logging.info('-done')
