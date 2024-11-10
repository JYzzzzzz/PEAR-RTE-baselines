#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import json
import os
from tqdm import tqdm
import re
# from IPython.core.debugger import set_trace
from pprint import pprint

from transformers import AutoModel, BertTokenizerFast, BertTokenizer, AutoTokenizer
import copy
import torch
import torch.nn as nn
# from torch.utils.data.distributed import DistributedSampler
# gpus = [4,5,6,7]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
import numpy
import logging
# from utils import Preprocessor,DefaultLogger
from common.utils import Preprocessor, DefaultLogger
from tplinker import (HandshakingTaggingScheme,
                      DataMaker4Bert,
                      DataMaker4BiLSTM,
                      TPLinkerBert,
                      TPLinkerBiLSTM,
                      MetricsCalculator)
# import wandb  # 可视化相关
import config as config_file
# from glove import Glove
import numpy as np


# In[ ]:


# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# train step
def train_step(model, batch_train_data, optimizer, loss_weights):
    if config["encoder"] == "BERT":
        # get field
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, \
        tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = batch_train_data
        """
                batch_ent_shaking_tag: (B, 1+2+...+seq_len)
                batch_head_rel_shaking_tag: (B, rel_num, 1+2+...+seq_len)
            """

        # to device
        batch_input_ids, batch_attention_mask, batch_token_type_ids, \
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                      batch_attention_mask.to(device),
                                      batch_token_type_ids.to(device),
                                      batch_ent_shaking_tag.to(device),
                                      batch_head_rel_shaking_tag.to(device),
                                      batch_tail_rel_shaking_tag.to(device)
                                      )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

        batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                          batch_ent_shaking_tag.to(device),
                                                                                                          batch_head_rel_shaking_tag.to(device),
                                                                                                          batch_tail_rel_shaking_tag.to(device)
                                                                                                          )

    # # 测试 metrics.get_rel_cpg 函数功能。正常
    # rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
    #                               batch_ent_shaking_tag,
    #                               batch_head_rel_shaking_tag,
    #                               batch_tail_rel_shaking_tag,
    #                               hyper_parameters["match_pattern"]
    #                               )
    # print(xxxxx)

    # zero the parameter gradients
    optimizer.zero_grad()

    # model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if config["encoder"] == "BERT":
    #     ent_shaking_outputs, head_rel_shaking_outputs, \
    #         tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
    #                                                  batch_attention_mask,
    #                                                  batch_token_type_ids,
    #                                                  )
    # elif config["encoder"] in {"BiLSTM", }:
    #     ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)
    if config["encoder"] == "BERT":
        ent_shaking_outputs, head_rel_shaking_outputs, \
        tail_rel_shaking_outputs = model(batch_input_ids,
                                         batch_attention_mask,
                                         batch_token_type_ids,
                                         )
        """
                ent_shaking_outputs: (B, 1+...+seq_len, 2)
                head_rel_shaking_outputs: (B, rel_num, 1+...+seq_len, 3)
            """
    elif config["encoder"] in {"BiLSTM", }:
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(batch_input_ids)

    # loss
    w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
    loss = w_ent * loss_func(ent_shaking_outputs, batch_ent_shaking_tag) \
           + w_rel * loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) \
           + w_rel * loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

    loss.backward()
    optimizer.step()

    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                 batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs,
                                                      batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs,
                                                      batch_tail_rel_shaking_tag)

    return loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item()


# valid step
def valid_step(model, batch_valid_data):
    # get batch field and send to device
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                                                                      batch_attention_mask.to(device),
                                                                                                                                                      batch_token_type_ids.to(device),
                                                                                                                                                      batch_ent_shaking_tag.to(device),
                                                                                                                                                      batch_head_rel_shaking_tag.to(device),
                                                                                                                                                      batch_tail_rel_shaking_tag.to(device)
                                                                                                                                                      )

    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

        batch_input_ids, batch_ent_shaking_tag, atch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (batch_input_ids.to(device),
                                                                                                         batch_ent_shaking_tag.to(device),
                                                                                                         batch_head_rel_shaking_tag.to(device),
                                                                                                         batch_tail_rel_shaking_tag.to(device)
                                                                                                         )

    # model predict
    # with torch.no_grad():
    #     if config["encoder"] == "BERT":
    #         ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
    #                                                                                                 batch_attention_mask,
    #                                                                                                 batch_token_type_ids,
    #                                                                                                 )
    #     elif config["encoder"] in {"BiLSTM", }:
    #         ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = rel_extractor(batch_input_ids)
    with torch.no_grad():
        if config["encoder"] == "BERT":
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(batch_input_ids,
                                                                                            batch_attention_mask,
                                                                                            batch_token_type_ids,
                                                                                            )
        elif config["encoder"] in {"BiLSTM", }:
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(batch_input_ids)

    # calculate score
    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                 batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs,
                                                      batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs,
                                                      batch_tail_rel_shaking_tag)

    rel_cpg, sample_list_pred = metrics.get_rel_cpg(sample_list, tok2char_span_list,
                                                    ent_shaking_outputs,
                                                    head_rel_shaking_outputs,
                                                    tail_rel_shaking_outputs,
                                                    hyper_parameters["match_pattern"]
                                                    )  # 返回correct的三元组个数，以及pred，gold个数

    return ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item(), rel_cpg, sample_list_pred


def train_epoch(model, dataloader, optimizer, scheduler, num_epoch, epoch_i):
    # train
    # rel_extractor.train()
    model.train()
    t_ep = time.time()
    start_lr = optimizer.param_groups[0]['lr']
    total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
    for batch_ind, batch_train_data in enumerate(dataloader):
        # print(f"  batch {batch_ind}")
        t_batch = time.time()
        z = (2 * len(rel2id) + 1)  # 2倍的关系
        steps_per_ep = len(dataloader)  # 有多少数据
        total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error #加速loss在一定的步数回归
        current_step = steps_per_ep * epoch_i + batch_ind  # ？
        w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)  # ？
        w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))  # ？
        loss_weights = {"ent": w_ent, "rel": w_rel}  # 给予不同任务的权重

        # train !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = train_step(
            model, batch_train_data, optimizer, loss_weights)
        scheduler.step()

        total_loss += loss
        total_ent_sample_acc += ent_sample_acc
        total_head_rel_sample_acc += head_rel_sample_acc
        total_tail_rel_sample_acc += tail_rel_sample_acc

        avg_loss = total_loss / (batch_ind + 1)
        avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)

        # 一个n个step打印一个
        if (batch_ind + 1) in [1, 10] or (batch_ind + 1) % 100 == 0:
            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, " \
                                 "train_loss: {}, t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, " \
                                 "t_tail_rel_sample_acc: {}, lr: {}, batch_time: {}, total_time: {} -------------"
            print(batch_print_format.format(experiment_name, config["run_name"],
                                            epoch_i + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            round(avg_loss, 4),
                                            round(avg_ent_sample_acc, 4),
                                            round(avg_head_rel_sample_acc, 4),
                                            round(avg_tail_rel_sample_acc, 4),
                                            '%.4g' % optimizer.param_groups[0]['lr'],
                                            round(time.time() - t_batch),
                                            round(time.time() - t_ep),
                                            ), end="")

        if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    if config["logger"] != "wandb":  # only log once for training if logger is not wandb
        logger.log({
            "train_loss": avg_loss,
            "train_ent_seq_acc": avg_ent_sample_acc,
            "train_head_rel_acc": avg_head_rel_sample_acc,
            "train_tail_rel_acc": avg_tail_rel_sample_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "time": time.time() - t_ep,
        })


def valid(model, dataloader):
    # valid
    # rel_extractor.eval()
    model.eval()

    t_ep = time.time()
    total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
    total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
    total_sample_list_pred = []
    for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
        # eval !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, \
        rel_cpg, batch_sample_list_pred = valid_step(model, batch_valid_data)

        total_ent_sample_acc += ent_sample_acc
        total_head_rel_sample_acc += head_rel_sample_acc
        total_tail_rel_sample_acc += tail_rel_sample_acc

        total_rel_correct_num += rel_cpg[0]
        total_rel_pred_num += rel_cpg[1]
        total_rel_gold_num += rel_cpg[2]

        total_sample_list_pred += batch_sample_list_pred

    avg_ent_sample_acc = total_ent_sample_acc / len(dataloader)
    avg_head_rel_sample_acc = total_head_rel_sample_acc / len(dataloader)
    avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(dataloader)

    # get p, r, f1 score
    rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)

    log_dict = {
        "val_ent_seq_acc": avg_ent_sample_acc,
        "val_head_rel_acc": avg_head_rel_sample_acc,
        "val_tail_rel_acc": avg_tail_rel_sample_acc,
        "val_prec": rel_prf[0],
        "val_recall": rel_prf[1],
        "val_f1": rel_prf[2],
        "time": time.time() - t_ep,
    }
    logger.log(log_dict)
    pprint(log_dict)
    rel_prf1 = tuple(round(val, 4) for val in rel_prf)

    return rel_prf1, total_sample_list_pred  # f1


def train_n_valid(model, dataloader_groups, optimizer, scheduler, num_epoch):
    train_dataloader, train_dataloader_for_eval, valid_dataloader, test_dataloader = dataloader_groups
    res_all = []
    for epoch_i in range(num_epoch):
        # print(f"training epoch {epoch_i + 1}")
        # train and eval !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        train_epoch(model, train_dataloader, optimizer, scheduler, num_epoch, epoch_i)  # 训练一个epoch

        print("\neval train")
        train_prf1, train_case = valid(model, train_dataloader_for_eval)  # 验证一次
        print("\neval valid")
        valid_prf1, valid_case = valid(model, valid_dataloader)  # 验证一次
        print("\neval test")
        test_prf1, test_case = valid(model, test_dataloader)  # 验证一次

        res_this_epoch = {
            'epoch': epoch_i + 1,
            'train': train_prf1,
            'valid': valid_prf1,
            'test': test_prf1,
        }
        res_all.append(res_this_epoch)
        res_all.sort(key=lambda x: x['valid'][-1], reverse=True)
        res_out = {'res_this_epoch': res_this_epoch, 'res_all': res_all}

        # SAVE
        global max_f1
        save_dir = os.path.join(model_state_dict_dir, f"checkpoint-epoch{epoch_i + 1}")
        if os.path.exists(save_dir) == 0:
            os.makedirs(save_dir)

        # save prediction case
        with open(os.path.join(save_dir, "prediction_train.json"), "w", encoding="utf-8") as fp:
            json.dump(train_case, fp, ensure_ascii=False, indent=4)
        with open(os.path.join(save_dir, "prediction_valid.json"), "w", encoding="utf-8") as fp:
            json.dump(valid_case, fp, ensure_ascii=False, indent=4)
        with open(os.path.join(save_dir, "prediction_test.json"), "w", encoding="utf-8") as fp:
            json.dump(test_case, fp, ensure_ascii=False, indent=4)
        with open(os.path.join(save_dir, "prediction_score.json"), "w", encoding="utf-8") as fp:
            json.dump(res_out, fp, ensure_ascii=False, indent=4)

        if valid_prf1[2] > config["f1_2_save"]:
            pass
            # torch.save(model.state_dict(),
            #            os.path.join(save_dir, "model_state_dict.pt"))

        # if valid_f1 >= max_f1:
        #     max_f1 = valid_f1
        #     if valid_f1 > config["f1_2_save"]:  # save the best model
        #         model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))  # 这段代码的作用是获取指定路径下以"model_state_dict_"为前缀，以".pt"为后缀的文件数量。
        #         torch.save(model.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(model_state_num)))
        # #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
        # #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num)))
        # print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


class Char_Token_SpanConverter(object):
    """
    用于数据集生成准确的 token_char_mapping, 并互化
    version 240725 : 考虑了span互化时，输入span为(x,x)的异常情况，print了一些提示信息。
    version 240825: 添加 返回mapping的函数
    """

    def __init__(self, tokenizer, add_special_tokens=False, has_return_offsets_mapping=True):
        """
        add_special_tokens: 如果 add_special_tokens=True，会将 [CLS] 考虑在内，token_span 数值整体+1
        has_return_offsets_mapping: bool. tokenizer是否包含return_offsets_mapping功能，若不包含，手动生成。
        """
        self.tokenizer = tokenizer
        self.token_info = None
        self.error_tok_spans = []  # {text, char_span, char_span_str, tok_span_str}
        self.add_special_tokens = add_special_tokens  # 不影响 tokenizer 初始化时设置的 add_special_tokens
        self.has_return_offsets_mapping = has_return_offsets_mapping

    def get_tok_span(self, text: str, char_span):

        # get mapping
        self._get_mapping(text)

        # get token span
        if char_span[0] == char_span[1]:
            token_span = self._get_tok_span((char_span[0], char_span[1] + 1))
            token_span = (token_span[0], token_span[0])
            print(f"\n-- Char_Token_SpanConverter.get_tok_span\n"
                  f"    get tok_span={token_span} by char_span={char_span} in \nsent={text}")
        else:  # normal situation
            token_span = self._get_tok_span(char_span)

        # # check
        # self._char_tok_span_check(char_span, token_span)
        return tuple(token_span)

    def get_char_span(self, text: str, token_span):
        # get mapping
        self._get_mapping(text)

        # get char span
        if token_span[0] == token_span[1]:
            char_span_list = self.token_info["tok2char_mapping"][token_span[0]:token_span[1] + 1]
            char_span = (char_span_list[0][0], char_span_list[0][0])
            print(f"\n-- Char_Token_SpanConverter.get_char_span\n"
                  f"    get char_span={char_span} by tok_span={token_span} in \nsent={text}")
        else:  # normal situation
            char_span_list = self.token_info["tok2char_mapping"][token_span[0]:token_span[1]]
            char_span = (char_span_list[0][0], char_span_list[-1][1])

        return char_span

    def get_mapping_tok2char(self, text):
        self._get_mapping(text)
        return self.token_info["tok2char_mapping"]  # 满足切片规则

    def get_mapping_char2tok(self, text):
        self._get_mapping(text)
        return self.token_info["char2tok_mapping"]

    def _get_mapping(self, text):
        """
        实际返回 encode_plus 生成的 token相关信息，其中添加了一些key，主要包括 char2tok_mapping
        """
        if self.token_info is not None and self.token_info["text"] == text:
            return  # 跳过重复操作

        if self.has_return_offsets_mapping is True:
            # Tokenizer 自带生成 offset_mapping(tok2char_mapping) 的功能
            token_info = self.tokenizer.encode_plus(text,
                                                    return_offsets_mapping=True,
                                                    add_special_tokens=self.add_special_tokens)
            token_info["text"] = text  # 添加原文
            token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])

            tok2char_span = token_info["offset_mapping"]
            token_info["tok2char_mapping"] = tok2char_span.copy()
            del token_info["offset_mapping"]

            char_num = None
            for tok_ind in range(len(tok2char_span) - 1, -1, -1):
                if tok2char_span[tok_ind][1] != 0:
                    char_num = tok2char_span[tok_ind][1]
                    break
            char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
            for tok_ind, char_sp in enumerate(tok2char_span):
                for char_ind in range(char_sp[0], char_sp[1]):
                    tok_sp = char2tok_span[char_ind]
                    # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                    if tok_sp[0] == -1:
                        tok_sp[0] = tok_ind
                    tok_sp[1] = tok_ind + 1
            token_info["char2tok_mapping"] = char2tok_span.copy()

        else:  # self.has_return_offsets_mapping is False
            token_info = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=self.add_special_tokens)
            token_info["text"] = text  # 添加原文
            token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])

            # ---------------------------------------- get char2tok_mapping
            tokens = token_info["tokens"].copy()
            char2tok_mapping = [(-1, -1)] * len(text)
            tokens_i = [0, 0]  # 起始：下标为0的token的下标为0的字符
            if tokens[0] == self.tokenizer.cls_token:
                tokens_i = [1, 0]  # 起始：下标为1的token的下标为0的字符
            # 遍历字符
            for c_i, c in enumerate(text):
                c_belong_unk = 0
                c_tokens = self.tokenizer.tokenize(c)
                if len(c_tokens) == 0:  # c 是一个空白字符
                    pass
                else:
                    ct = c_tokens[0]
                    # 查找字符在哪个token中
                    while ct not in tokens[tokens_i[0]]:
                        if tokens[tokens_i[0]] == '[UNK]' and ct not in tokens[tokens_i[0] + 1]:
                            c_belong_unk = 1
                            break
                        tokens_i[0] += 1
                        tokens_i[1] = 0
                        assert tokens_i[0] < len(tokens), f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
                    if ct == '[UNK]':
                        c_belong_unk = 1

                    if c_belong_unk == 0:
                        # 查找字符在token中哪个位置
                        ct_pos = tokens[tokens_i[0]].find(ct, tokens_i[1])
                        assert ct_pos >= tokens_i[1], f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
                        # 添加到char2tok_mapping
                        char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
                        # 更新tokens_i
                        tokens_i[1] = ct_pos + len(ct)
                        if tokens_i[1] >= len(tokens[tokens_i[0]]):
                            tokens_i[0] += 1
                            tokens_i[1] = 0
                    else:
                        char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
            token_info["char2tok_mapping"] = char2tok_mapping.copy()

            # ---------------------------------------- get tok2char_mapping
            tok2char_mapping = [(-1, -1)] * len(tokens)
            for c_i in range(len(text)):
                if char2tok_mapping[c_i][0] == -1 or char2tok_mapping[c_i][0] == char2tok_mapping[c_i][1]:
                    continue
                token_i = char2tok_mapping[c_i][0]
                if tok2char_mapping[token_i] == (-1, -1):
                    tok2char_mapping[token_i] = (c_i, c_i + 1)
                else:
                    assert c_i + 1 > tok2char_mapping[token_i][1]
                    tok2char_mapping[token_i] = (tok2char_mapping[token_i][0], c_i + 1)
            token_info["tok2char_mapping"] = tok2char_mapping.copy()

        self.token_info = token_info
        # return token_info

    def _get_tok_span(self, char_span):
        """
        得到 tok_span
        """
        # char2tok_span: 列表，每个元素表示每个句中字符对应的token下标。
        #   每个元素一般取值为[a,a+1]，
        #   如果连续多个元素位于一个token中，则会出现`[a,a+1],[a,a+1],...`，
        #   如果是例如空格等字符，不会出现在token中，则取值[-1,-1]

        tok_span_list = self.token_info["char2tok_mapping"][char_span[0]:char_span[1]]
        tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
        return tok_span

    def _char_tok_span_check(self, char_span, tok_span):
        """
        校验 tok_span 是否能抽取出与 char_span 一样的文本
        token_info: 必须包含 text, input_ids
        tokenizer: 必须是生成 token_info 的 tokenizer
        char_span: 长度为2的列表或元组，暂时不考虑分段情况
        tok_span: 长度为2的列表或元组，暂时不考虑分段情况
        """
        sub_text_from_char0 = self.token_info['text'][char_span[0]:char_span[1]]
        sub_text_from_char = self.tokenizer.decode(self.tokenizer.encode(sub_text_from_char0, add_special_tokens=False))

        sub_text_from_token = self.tokenizer.decode(self.token_info['input_ids'][tok_span[0]:tok_span[1]])

        if sub_text_from_char == sub_text_from_token:
            return True
        else:
            error_tok_span = {
                'text': self.token_info['text'],
                'char_span': char_span,
                'char_span_str': sub_text_from_char,
                'tok_span_str': sub_text_from_token
            }
            if error_tok_span not in self.error_tok_spans:
                self.error_tok_spans.append(error_tok_span)
                print(f"char_span string: [{sub_text_from_char0}][{sub_text_from_char}], but tok_span string: [{sub_text_from_token}]")
            return False


def adjust_for_cmim(datas, tokenizer):
    """
    transform the format of CMIM23-NOM1-RA

    input:
        {
            "id": 0,
            "text": "系统消息4包含LAI、RACH控制参数信息。",
            "relation_list": [
                {
                    "subject": "系统消息4",
                    "predicate": "含有",
                    "object": "LAI控制参数信息",
                    "subj_char_span": [ [ 0, 5 ] ],
                    "obj_char_span": [ [ 7, 10 ], [ 15, 21 ] ]
                },
                ...
            ]
        },
    output:
        {
            "id": 0,
            "text": "系统消息4包含LAI、RACH控制参数信息。",
            "relation_list": [
                {
                    "subject": "系统消息4",
                    "predicate": "含有",
                    "object": "RACH控制参数信息",
                    "subj_char_span": [ 0, 5 ],
                    "obj_char_span": [ 11, 21 ],
                    "subj_tok_span": [ 0, 5 ],
                    "obj_tok_span": [ 9, 16 ]
                },
                ...
            ],
            "entity_list": [
                {
                    "text": "系统消息4",
                    "type": "Default",
                    "char_span": [ 0, 5 ],
                    "tok_span": [ 0, 5 ]
                },
                ...
            ]
        },

    operations:
        1. delete all triples with segmented entities
        2. change the type of "subj_char_span" and "obj_char_span"
        3. add "subj_tok_span" and "obj_tok_span"
        4. add "entity_list"
    """

    span_converter = Char_Token_SpanConverter(tokenizer)

    datas_out = []

    for data in datas:
        id_, text, triples = data['id'], data['text'], data['relation_list']

        triples_out = []
        entities_out = []
        for triple in triples:
            assert type(triple) == dict, f"\n{triples}"
            if len(triple['subj_char_span']) > 1 or len(triple['obj_char_span']) > 1:
                continue  # delete all triples with segmented entities

            triple_out = {
                "subject": triple['subject'],
                "predicate": triple['predicate'],
                "object": triple['object'],
                'subj_char_span': triple['subj_char_span'][0].copy(),
                'obj_char_span': triple['obj_char_span'][0].copy()
            }
            subj_tok_span = span_converter.get_tok_span(text, triple_out['subj_char_span'])
            obj_tok_span = span_converter.get_tok_span(text, triple_out['obj_char_span'])
            triple_out['subj_tok_span'] = list(subj_tok_span)
            triple_out['obj_tok_span'] = list(obj_tok_span)
            if triple_out not in triples_out:
                triples_out.append(triple_out.copy())

            subj_out = {
                "text": triple_out['subject'],
                "type": "Default",
                "char_span": triple_out['subj_char_span'].copy(),
                "tok_span": triple_out['subj_tok_span'].copy()
            }
            if subj_out not in entities_out:
                entities_out.append(subj_out.copy())
            obj_out = {
                "text": triple_out['object'],
                "type": "Default",
                "char_span": triple_out['obj_char_span'].copy(),
                "tok_span": triple_out['obj_tok_span'].copy()
            }
            if obj_out not in entities_out:
                entities_out.append(obj_out.copy())

        data_out = {
            'id': id_,
            'text': text,
            'relation_list': triples_out,
            'entity_list': entities_out,
        }
        datas_out.append(data_out.copy())

    return datas_out


def main():   # 只是为了方便指示程序开始位置
    pass


if __name__ == '__main__':

    # -------------------------------------------------- config
    print("config ...")

    # time.sleep(5)
    config = config_file.train_config  # 来自 config.py
    hyper_parameters = config["hyper_parameters"]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    time.sleep(5)

    # for reproductivity
    torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
    torch.backends.cudnn.deterministic = True

    data_home = config["data_home"]
    experiment_name = config["exp_name"]
    train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
    # train_data_path = os.path.join(data_home, experiment_name, config["train_test_data"])
    valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
    test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
    rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])

    if config["logger"] == "wandb":
        pass
        # # init wandb
        # wandb.init(project=experiment_name,
        #            name=config["run_name"],
        #            config=hyper_parameters  # Initialize config
        #            )
        #
        # wandb.config.note = config["note"]
        #
        # model_state_dict_dir = wandb.run.dir
        # logger = wandb
    else:
        logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
        model_state_dict_dir = config["path_to_save_model"]
        if not os.path.exists(model_state_dict_dir):
            os.makedirs(model_state_dict_dir)

    print(f"config = {config}")
    # time.sleep(5)
    """
    config = {
        'train_data': 'train_data.json', 'valid_data': 'valid_data.json', 
        'train_test_data': 'v_test.json', 'rel2id': 'rel2id.json', 
        'logger': 'wandb', 'f1_2_save': 0, 'fr_scratch': True, 'note': 'start from scratch', 
        'model_state_dict_path': '', 
        'hyper_parameters': {
            'shaking_type': 'cat', 'inner_enc_type': 'lstm', 'dist_emb_size': -1, 
            'ent_add_dist': False, 'rel_add_dist': False, 'match_pattern': 'only_head_text', 
            'lr': 5e-05, 'batch_size': 25, 'epochs': 50, 'seed': 2333, 'log_interval': 10, 
            'max_seq_len': 100, 'sliding_len': 20, 'loss_weight_recover_steps': 6000, 
            'scheduler': 'CAWR', 'T_mult': 1, 'rewarm_epoch_num': 2
        }, 
        'exp_name': 'baidu_relation', 'encoder': 'BERT', 'run_name': 'TP1+cat+BERT', 
        'data_home': '/home/yuanchaoyi/TPlinker-joint-extraction/data4bert', 
        'bert_path': '/home/yuanchaoyi/BeiKe/QA_match/roberta_base'
    }
    """

    # -------------------------------------------------- Load Data
    print("load data ...")
    # time.sleep(5)
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    # train_test_data = json.load(open(train_test_path,'r',encoding='utf8'))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    test_data = json.load(open(test_data_path, "r", encoding="utf-8"))
    ##### train_data_path = data_home + exp_name + train_data

    # valid_data = valid_data[:10]

    # # Split

    # @specific
    if config["encoder"] == "BERT":
        # tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
        tokenizer = BertTokenizerFast.from_pretrained(
            config["bert_path"], add_special_tokens=False, do_lower_case=False)
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]  # 偏移量
    elif config["encoder"] in {"BiLSTM", }:
        tokenize = lambda text: text.split(" ")


        def get_tok2char_span_map(text):
            the_tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in the_tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    # 将数据集转换为训练验证所需的格式 jyz chg 2409
    print("\n-- adjust the format of dataset")
    test_data = adjust_for_cmim(test_data, tokenizer)
    print(test_data[0])
    time.sleep(10)
    valid_data = adjust_for_cmim(valid_data, tokenizer)
    train_data = adjust_for_cmim(train_data, tokenizer)

    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)  # 构建处理的函数

    # train and valid max token num
    max_tok_num = 0
    all_data = train_data + valid_data + test_data

    for i, sample in enumerate(all_data):
        tokens = tokenize(sample["text"])
        if i < 3:
            print(f"text=\n{sample['text']}\ntokens=\n{tokens}")
        max_tok_num = max(max_tok_num, len(tokens))
    print(f"max_tok_num = {max_tok_num}")  # 获取句子的最大长度

    if max_tok_num > hyper_parameters["max_seq_len"]:  # 截断长度。并使用滑动窗口策略增加新样本
        train_data = preprocessor.split_into_short_samples(train_data,
                                                           hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=config["encoder"]  # 超过长度则滑动窗口得到新的样本
                                                           )
        # print(train_data[:5])
        """
        [
            {
                'id': 0, 
                'text': '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名
                        小说电视剧《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说《裸婚
                        时代》是月影兰析创作的一部情感小说《琅琊榜》是', 
                'tok_offset': 0, 
                'char_offset': 0, 
                'entity_list': [{'text': '步步惊心', 'type': '图书作品', 'char_span': [1, 5], 'tok_span': [1, 5]}, ...], 
                'relation_list': [{'subject': '步步惊心', 'object': '桐华', 'subj_char_span': [1, 5], 
                                    'obj_char_span': [13, 15], 'predicate': '作者', 
                                    'subj_tok_span': [1, 5], 'obj_tok_span': [13, 15]}, ...]
            }, 
            {
                'id': 0, 
                'text': '小说《甄嬛传》改编自流潋紫所著的同名小说电视剧《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说《裸婚时代》是月影兰析创作的一部情感小说《琅琊榜》是根据海宴同名网络小说改编电视剧《宫锁心玉', 
                'tok_offset': 20, 
                'char_offset': 20, 
                'entity_list': [{'text': '甄嬛传', 'type': '图书作品', 'char_span': [3, 6], 'tok_span': [3, 6]}, ...], 
                'relation_list': [{'subject': '甄嬛传', 'object': '流潋紫', 'subj_char_span': [3, 6], 'obj_char_span': [10, 13], 'predicate': '作者', 'subj_tok_span': [3, 6], 'obj_tok_span': [10, 13]}, ...]
            }, 
            ...
        ]
        """
        valid_data = preprocessor.split_into_short_samples(valid_data,
                                                           hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=config["encoder"]
                                                           )
        test_data = preprocessor.split_into_short_samples(test_data,
                                                          hyper_parameters["max_seq_len"],
                                                          sliding_len=hyper_parameters["sliding_len"],
                                                          encoder=config["encoder"]
                                                          )

    print("train: {}".format(len(train_data)),
          "valid: {}".format(len(valid_data)),
          "test: {}".format(len(test_data)),
          )

    # # Tagger (Decoder)

    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])  # max_len 长度
    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)  # 初始化
    ##### tplinker.py中，tag策略的相关操作

    # # Dataset

    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
        data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)  # (sample,input_ids，attention_mask,token_type_ids,tok2char_span,spots_tuple,)
        ##### DataMaker4Bert: tplinker.py中的dataloader

    elif config["encoder"] in {"BiLSTM", }:
        token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
        token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
        idx2token = {idx: tok for tok, idx in token2idx.items()}


        def text2indices(text, max_seq_len):
            input_ids = []
            the_tokens = text.split(" ")
            for tok in the_tokens:
                if tok not in token2idx:
                    input_ids.append(token2idx['<UNK>'])
                else:
                    input_ids.append(token2idx[tok])
            if len(input_ids) < max_seq_len:
                input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
            input_ids = torch.tensor(input_ids[:max_seq_len])
            return input_ids


        data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)

    indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)  # 获取输入
    # index_train_data = data_maker.get_indexed_data(train_test_data,max_seq_len)
    indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len)
    """ print(indexed_train_data[0])
    (
        {'id': 0, 'text': '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名小说电视剧
            《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说《裸婚时代》是月影兰析创作的一部情感小说
            《琅琊榜》是', 'tok_offset': 0, 'char_offset': 0, 
            'entity_list': [{'text': '顾漫', 'type': '人物', 'char_span': [53, 55],  'tok_span': [53, 55]}, 
            {'text': '何以笙箫默', 'type': '图书作品', 'char_span': [44, 49], 'tok_span': [44, 49]}, 
            {'text': '桐华', 'type': '人物', 'char_span': [13, 15], 'tok_span': [13, 15]}, 
            {'text': '步步惊心', 'type': '图书作品', 'char_span': [1, 5], 'tok_span': [1, 5]}, 
            {'text': '流潋紫', 'type': '人物', 'char_span': [30, 33], 'tok_span': [30, 33]}, 
            {'text': '甄嬛传', 'type': '图书作品', 'char_span': [23, 26], 'tok_span': [23, 26]}, 
            {'text': 'fresh果果', 'type': '人物', 'char_span': [67, 74], 'tok_span': [67, 70]}, 
            {'text': '花千骨', 'type': '图书作品', 'char_span': [60, 63], 'tok_span': [60, 63]}, 
            {'text': '月影兰析', 'type': '人物', 'char_span': [85, 89], 'tok_span': [81, 85]}, 
            {'text': '裸婚时代', 'type': '图书作品', 'char_span': [79, 83], 'tok_span': [75, 79]}, 
            {'text': '琅琊榜', 'type': '图书作品', 'char_span': [99, 102], 'tok_span': [95, 98]}], 
            'relation_list': [{'subject': '何以笙箫默', 'object': '顾漫', 'subj_char_span': [44, 49], 
            'obj_char_span': [53, 55], 'predicate': '作者', 'subj_tok_span': [44, 49], 
            'obj_tok_span': [53, 55]}, {'subject': '步步惊心', 'object': '桐华', 
            'subj_char_span': [1, 5], 'obj_char_span': [13, 15], 'predicate': '作者', 
            'subj_tok_span': [1, 5], 'obj_tok_span': [13, 15]}, {'subject': '甄嬛传', 
            'object': '流潋紫', 'subj_char_span': [23, 26], 'obj_char_span': [30, 33], 
            'predicate': '作者', 'subj_tok_span': [23, 26], 'obj_tok_span': [30, 33]}, 
            {'subject': '花千骨', 'object': 'fresh果果', 'subj_char_span': [60, 63], 
            'obj_char_span': [67, 74], 'predicate': '作者', 'subj_tok_span': [60, 63], 
            'obj_tok_span': [67, 70]}, {'subject': '裸婚时代', 'object': '月影兰析', 
            'subj_char_span': [79, 83], 'obj_char_span': [85, 89], 'predicate': '作者', 
            'subj_tok_span': [75, 79], 'obj_tok_span': [81, 85]}]}, 
        
        tensor([  517,  3635,  3635,  2661,  2552,   518,  3121,  5356,  5632,  5865,
             1399,   868,  2157,  3432,  1290,  4638,  1398,  1399,  3926,  4959,
             2207,  6432,   517,  4488,  2083,   837,   518,  3121,  5356,  5632,
             3837,  4046,  5166,  2792,  5865,  4638,  1398,  1399,  2207,  6432,
             4510,  6228,  1196,   517,   862,   809,  5012,  5054,  7949,   518,
             3121,  5356,  5632,  7560,  4035,  1398,  1399,  2207,  6432,   517,
             5709,  1283,  7755,   518,  3121,  5356,  5632, 12718,  3362,  3362,
             1398,  1399,  2207,  6432,   517,  6180,  2042,  3198,   807,   518,
             3221,  3299,  2512,  1065,  3358,  1158,   868,  4638,   671,  6956,
             2658,  2697,  2207,  6432,   517,  4414,  4418,  3528,   518,  3221]),     # token id
        tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1]),    # 应该是attention mask
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0]),    # 应该是bert所需的segment
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), 
            (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), 
            (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), 
            (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), 
            (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), 
            (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), 
            (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), 
            (65, 66), (66, 67), (67, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), 
            (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), 
            (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), 
            (96, 97), (97, 98), (98, 99), (99, 100), (100, 101), (101, 102), (102, 103), 
            (103, 104)],   # 应该是每个token对应的char位置，注意看67
        (
            [(44, 48, 1), (53, 54, 1), (1, 4, 1), (13, 14, 1), (23, 25, 1), (30, 32, 1), (60, 62, 1), 
                (67, 69, 1), (75, 78, 1), (81, 84, 1)],   # 实体标注
            [(10, 44, 53, 1), (10, 1, 13, 1), (10, 23, 30, 1), (10, 60, 67, 1), (10, 75, 81, 1)],    
                # 关系标注，主客体起始token，(rel2id, 前实体起始, 后实体起始, 前实体是主体还是客体)
            [(10, 48, 54, 1), (10, 4, 14, 1), (10, 25, 32, 1), (10, 62, 69, 1), (10, 78, 84, 1)] 
                # 关系标注，主客体结束token，(rel2id, 前实体结束, 后实体结束, 前实体是主体还是客体)
        )
    )
    """

    train_dataloader = DataLoader(MyDataset(indexed_train_data),
                                  batch_size=hyper_parameters["batch_size"],
                                  shuffle=True,
                                  collate_fn=data_maker.generate_batch,
                                  )
    train_dataloader_for_eval = DataLoader(MyDataset(indexed_train_data[:200]),
                                           batch_size=hyper_parameters["batch_size"],
                                           shuffle=False,
                                           collate_fn=data_maker.generate_batch,
                                           )
    valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                                  batch_size=hyper_parameters["batch_size"],
                                  shuffle=False,
                                  collate_fn=data_maker.generate_batch,
                                  )
    test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                 batch_size=hyper_parameters["batch_size"],
                                 shuffle=False,
                                 collate_fn=data_maker.generate_batch,
                                 )
    dataloader_groups = (train_dataloader, train_dataloader_for_eval, valid_dataloader, test_dataloader)

    # # have a look at dataloader
    # train_data_iter = iter(train_dataloader)
    # batch_data = next(train_data_iter)
    # text_id_list, text_list, batch_input_ids, \
    # batch_attention_mask, batch_token_type_ids, \
    # offset_map_list, batch_ent_shaking_tag, \
    # batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_data

    # print(text_list[0])
    # print()
    # print(tokenizer.decode(batch_input_ids[0].tolist()))
    # print(batch_input_ids.size())
    # print(batch_attention_mask.size())
    # print(batch_token_type_ids.size())
    # print(len(offset_map_list))
    # print(batch_ent_shaking_tag.size())
    # print(batch_head_rel_shaking_tag.size())
    # print(batch_tail_rel_shaking_tag.size())

    # -------------------------------------------------- Model
    print("prepare model ...")
    # time.sleep(5)

    if config["encoder"] == "BERT":
        encoder = AutoModel.from_pretrained(config["bert_path"])
        hidden_size = encoder.config.hidden_size
        fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
        rel_extractor = TPLinkerBert(encoder,
                                     len(rel2id),
                                     hyper_parameters["shaking_type"],
                                     hyper_parameters["inner_enc_type"],
                                     hyper_parameters["dist_emb_size"],
                                     hyper_parameters["ent_add_dist"],
                                     hyper_parameters["rel_add_dist"],
                                     )

    # elif config["encoder"] in {"BiLSTM", }:
    #     glove = Glove()
    #     glove = glove.load(config["pretrained_word_embedding_path"])
    #
    #     # prepare embedding matrix
    #     word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
    #     count_in = 0
    #
    #     # 在预训练词向量中的用该预训练向量
    #     # 不在预训练集里的用随机向量
    #     for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
    #         if tok in glove.dictionary:
    #             count_in += 1
    #             word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]
    #
    #     print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token))) # 命中预训练词向量的比例
    #     word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)
    #
    #     fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hyper_parameters["dec_hidden_size"]]).to(device)
    #     rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix,
    #                                    hyper_parameters["emb_dropout"],
    #                                    hyper_parameters["enc_hidden_size"],
    #                                    hyper_parameters["dec_hidden_size"],
    #                                    hyper_parameters["rnn_dropout"],
    #                                    len(rel2id),
    #                                    hyper_parameters["shaking_type"],
    #                                    hyper_parameters["inner_enc_type"],
    #                                    hyper_parameters["dist_emb_size"],
    #                                    hyper_parameters["ent_add_dist"],
    #                                    hyper_parameters["rel_add_dist"],
    #                                   )

    rel_extractor = rel_extractor.to(device)


    # In[ ]:

    # all_paras = sum(x.numel() for x in rel_extractor.parameters())
    # enc_paras = sum(x.numel() for x in encoder.parameters())

    # In[ ]:

    # print(all_paras, enc_paras)
    # print(all_paras - enc_paras)

    # # Metrics

    # In[ ]:

    def bias_loss(weights=None):
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device)
        cross_en = nn.CrossEntropyLoss(weight=weights)
        return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


    loss_func = bias_loss()  # a function

    # In[ ]:

    metrics = MetricsCalculator(handshaking_tagger)

    # # Train

    # In[ ]:

    # In[ ]:

    max_f1 = 0.

    # In[ ]:

    print("prepare optimizer ...")
    # time.sleep(5)

    # optimizer
    init_learning_rate = float(hyper_parameters["lr"])  # 学习率
    optimizer = torch.optim.AdamW(rel_extractor.parameters(), lr=init_learning_rate)

    if hyper_parameters["scheduler"] == "CAWR":  # 退火
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)

    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    # In[ ]:

    if not config["fr_scratch"]:
        model_state_path = config["model_state_dict_path"]
        rel_extractor.load_state_dict(torch.load(model_state_path))
        print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))
    # rel_extractor = rel_extractor

    # train and valid !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("start train ...")
    # time.sleep(5)
    train_n_valid(rel_extractor, dataloader_groups, optimizer, scheduler, hyper_parameters["epochs"])
