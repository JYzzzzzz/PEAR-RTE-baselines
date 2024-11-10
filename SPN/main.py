import argparse, os, torch
import random
import time
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
from transformers import BertTokenizer


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    data_arg = add_argument_group('Data')
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-exact")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/exact_data/train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/exact_data/valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/exact_data/test.json")
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-partial")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/casrel_data/new_train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/casrel_data/new_valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/casrel_data/new_test.json")

    # data_arg.add_argument('--dataset_name', type=str, default="WebNLG")
    # data_arg.add_argument('--train_file', type=str, default="./data/WebNLG/clean_WebNLG/new_train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/WebNLG/clean_WebNLG/new_valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/WebNLG/clean_WebNLG/new_test.json")
    data_arg.add_argument('--dataset_name', type=str, default="CMIM23-NOM1-RA")   # 仅用于输出文件名、路径名的生成，作用不大，可改动
    data_arg.add_argument('--train_file', type=str, default="./data/CMIM23-NOM1-RA/train_data.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/CMIM23-NOM1-RA/valid_data.json")
    data_arg.add_argument('--test_file', type=str, default="./data/CMIM23-NOM1-RA/test_data.json")
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    # data_arg.add_argument('--bert_directory', type=str, default="./bert_base_cased/")
    data_arg.add_argument('--bert_directory', type=str,
        default="./pretrained/chinese-bert-wwm-ext")
    data_arg.add_argument("--partial", type=str2bool, default=False)
    data_arg.add_argument('--output_dir', type=str, default="./outputs/240710/")

    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
    learn_arg.add_argument('--num_generated_triples', type=int, default=15)  # !!!!!
    learn_arg.add_argument('--num_decoder_layers', type=int, default=3)
    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    learn_arg.add_argument('--na_rel_coef', type=float, default=0.25)    # !!!!! 应该是无关类某种权重
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
    learn_arg.add_argument('--batch_size', type=int, default=8)    # !!!!!
    learn_arg.add_argument('--max_epoch', type=int, default=100)    # !!!!!
    learn_arg.add_argument('--max_text_token_len', type=int, default=200)    # !!!!!
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=0)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])

    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=100)
    evaluation_arg.add_argument('--max_span_length', type=int, default=30)  # !!!!! NYT webNLG 10

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=0)
    misc_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    if not torch.cuda.is_available():
        args.use_gpu = False
        print("use cpu")
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)

    # jyz chg 2409
    # args.decoder_lr = args.encoder_lr*2
    print(f"-- args.encoder_lr={args.encoder_lr}, args.decoder_lr={args.decoder_lr}")
    time.sleep(5)

    tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=True)
    data = build_data(args, tokenizer=tokenizer)
    model = SetPred4RE(args, data.relational_alphabet.size())  # model
    trainer = Trainer(model, data, args, tokenizer=tokenizer)
    trainer.train_model()

    print("-- relations:")
    print(data.relational_alphabet.instance2index)
    print(data.relational_alphabet.instances)
    print("")


"""
cd disk_vdb2_512G/jyz_projects/SPN/SPN4RE-main/
python -m main --batch_size 16 --output_dir outputs/240710_batchsize16
"""
