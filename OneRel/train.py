import config
import framework
import argparse
import models
import os, time
import torch
import numpy as np
import random


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dropout_prob', type=float, default=0.2)
parser.add_argument('--entity_pair_dropout', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--bert_max_len', type=int, default=400)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--eval_step', type=int, default=500)
parser.add_argument('--gpu_id', type=str, default="0")

parser.add_argument('--model_name', type=str, default='OneRel', help='name of the model')
parser.add_argument('--multi_gpu', type=bool, default=False)
# parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--dataset', type=str, default='CMIM23-NOM1-RA')
parser.add_argument('--train_prefix', type=str, default='train_data')  # train_triples
parser.add_argument('--dev_prefix', type=str, default='dev_data')  # dev_triples
parser.add_argument('--test_prefix', type=str, default='test_data')  # test_triples
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--rel_num', type=int, default=13)  # relation type number. 24 origin
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--output_dir', type=str, default="./outputs/240628_len150")
args = parser.parse_args()

con = config.Config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

seed = 2179
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# jyz chg 2406. just a note: the code for finding token span is in data_loader.REDataset.__getitem__
# tokenizer init in "data_loader.py - class ZhTokenizer - def init"
print("-- fw = framework.Framework(con)")
fw = framework.Framework(con)

print("-- models.RelModel init")
model = {
    'OneRel': models.RelModel
}

print("-- fw.train(model[args.model_name])")
fw.train(model[args.model_name])

"""
conda activate python3_10
cd disk_vdb2_512G/jyz_projects/OneRel/OneRel_chinese-main/
python train.py
"""

