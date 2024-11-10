import argparse
from main import *
import torch

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--cuda_id', default="0", type=str)
parser.add_argument('--base_path', default="./datasets", type=str)
# parser.add_argument('--dataset', default='WebNLG', type=str)
parser.add_argument('--dataset', default='CMIM23-NOM1-RA', type=str)  # 仅用于路径组成
# parser.add_argument('--train', default="train", type=str).json
parser.add_argument('--train', default="train", type=str)   # 这个不是文件名，只是一个标志
parser.add_argument('--bert_learning_rate', default=3e-5, type=float)
parser.add_argument('--other_learning_rate', default=(3e-5)*5, type=float)
parser.add_argument('--num_train_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=6, type=int)

parser.add_argument('--max_len', default=200, type=int)    # 最长长度 原100
parser.add_argument('--train_segment_entity_strategy', default='longest_slice',
                    type=str, choices=['longest_slice', 'delete'])
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--bert_vocab_path', default="./pretrained/chinese-bert-wwm-ext/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="./pretrained/chinese-bert-wwm-ext/config.json", type=str)
parser.add_argument('--bert_model_path', default="./pretrained/chinese-bert-wwm-ext/pytorch_model.bin", type=str)

# output
parser.add_argument('--file_id', default='999', type=str)
parser.add_argument('--output_dir', default='outputs/240825/', type=str)

args = parser.parse_args()

if args.train=="train":
    train(args)
else:
    test(args)


