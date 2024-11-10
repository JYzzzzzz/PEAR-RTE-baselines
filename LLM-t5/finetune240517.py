
import argparse
import functools  # 用于需要将函数名赋值给某个参数，同时函数名也有输入参数时
import json
import os
import subprocess
import threading
import time

import numpy as np
import yaml
from datasets import Dataset
from rouge_chinese import Rouge  # 一个评估指标相关的库，指标称为rouge
from tqdm import tqdm
from transformers import AutoConfig, T5Tokenizer, \
    T5ForConditionalGeneration, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


class Histogram:
    """
    直方图相关类

    1、初始化
    2、使用 input_one_data 一个一个添加数据
    3、使用 get_statistic_result 输出统计数据

    version:
        -- 240908: 添加 get_statistic_result 方法
    """

    def __init__(self, left_lim, right_lim, interval, init_show: str = ""):
        """

        :param left_lim: 统计的左边界
        :param right_lim: 统计的右边界
        :param interval: 各区间的间隔。边界规则：[)，最后一个区间规则：[]
        :param init_show: 没啥用
        """
        self.statistic_info = []
        self.statistic_info_simple = []  # 直接显示这个即可
        left = left_lim  # 每一柱的左边界
        while left < right_lim:
            right = right_lim if left + interval >= right_lim else left + interval
            col_info = [left, right, 0, 0.]  # 左边界，右边界，个数，占比。!!!!!!!!!!!!!
            # 边界规则：[)，最后一个区间规则：[]
            col_info_simple = [round(left, 2), 0.]  # 左边界，占比
            self.statistic_info.append(col_info.copy())
            self.statistic_info_simple.append(col_info_simple.copy())
            left = right
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.sample_in_lim_num = 0
        self.larger_num = 0
        self.smaller_num = 0
        # print("-- a histogram has been initialized: {}".format(init_show))
        # print(self.statistic_info_simple)

    def input_one_data(self, data):  # 直方图统计时添加一个数据
        if data < self.left_lim:
            self.smaller_num += 1
            return
        elif data > self.right_lim:
            self.larger_num += 1
            return

        for i in range(len(self.statistic_info) - 1, -1, -1):
            if self.statistic_info[i][0] <= data <= self.statistic_info[i][1]:  # [l, r)
                self.statistic_info[i][2] += 1
                break

    def update_ratio(self):  # 直方图显示前更新比率
        sample_num = 0
        for col_info in self.statistic_info:
            sample_num += col_info[2]
        self.sample_in_lim_num = sample_num

        if sample_num <= 0:  # 防止零除错误
            sample_num = 1

        for i in range(len(self.statistic_info)):
            self.statistic_info[i][3] = float(self.statistic_info[i][2]) / sample_num
            self.statistic_info_simple[i][1] = round(self.statistic_info[i][3], 2)

    def get_statistic_result(self, simple=True):
        """
        获取直方图统计数据
        :param simple: 返回的是简要数据还是完整数据
                        统计数据简要数据格式：[左边界，占比]
                        统计数据完整数据格式：[左边界，右边界，个数，占比]
        :return: 统计数据 list[list]
        """
        self.update_ratio()

        if simple:
            output = [["(-inf, l_lim)", float(self.smaller_num) / self.sample_in_lim_num]] + \
                     self.statistic_info_simple + \
                     [["(r_lim, inf)", float(self.larger_num) / self.sample_in_lim_num]]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    if type(output[i][j]) not in [str, int]:  # float
                        output[i][j] = round(output[i][j], 2)
            return output
        else:
            output = [["(-inf, l_lim)", self.smaller_num, float(self.smaller_num) / self.sample_in_lim_num]] + \
                     self.statistic_info + \
                     [["(r_lim, inf)", self.larger_num, float(self.larger_num) / self.sample_in_lim_num]]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    if type(output[i][j]) not in [str, int]:  # float
                        output[i][j] = round(output[i][j], 4)
            return output


def suffix_find(input_str, symbol_l_str):
    """
    find the suffix sub-string after the last `symbol_l_str`

    """
    symbol_l_pos = input_str.find(symbol_l_str)  # find the position of left boundary symbol of span
    if symbol_l_pos < 0:
        return ""
    sub_pos_l = symbol_l_pos + len(symbol_l_str)

    sub_str = input_str[sub_pos_l:]

    symbol_l_pos2 = sub_str.find(symbol_l_str)
    while symbol_l_pos2 > -1:
        # 截掉前缀，确保子串中没有span_l_str
        sub_pos_l += symbol_l_pos2 + len(symbol_l_str)
        sub_str = sub_str[symbol_l_pos2+len(symbol_l_str):]
        symbol_l_pos2 = sub_str.find(symbol_l_str)

    return sub_str


def compute_metric(evalPred, tokenizer):
    # 计算生成答案指标的函数，指标为rouge
    predictions, labels = evalPred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # if len(decode_preds) == 0:  # 为了防止 pred_tokens 为空
    #     decode_preds = ['-+-+']
    # if len(decode_labels) == 0:  # 为了防止 pred_tokens 为空
    #     decode_labels = ['-+-+']

    # type of decode_preds should be list[list[str]]
    decode_preds = [" ".join(p) for p in decode_preds]
    decode_labels = [" ".join(l) for l in decode_labels]
    decode_preds = [item+' .' for item in decode_preds]   # 强行新增一个字符，防止为空
    decode_labels = [item+' .' for item in decode_labels]
    rouge = Rouge()
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
    # return scores
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


# def process_func(samples):
#     # example 只有 title、content 两个键
#     contents = ["摘要生成: \n" + e for e in exmaples["content"]]
#     inputs = tokenizer(contents, max_length=384, truncation=True)
#     labels = tokenizer(text_target=exmaples["title"], max_length=64, truncation=True)
#     inputs["labels"] = labels["input_ids"]
#     return inputs


def dataset_load(samples, tokenizer, args):
    """

    :param samples: samples[?] = {
                "ques": "",
                "ans": ""
            },
    :param tokenizer:
    :param args:
    :return: format = {
                'input_ids': [[], [], ...],
                'attention_mask': [[], [], ...],
                'labels': [[], [], ...],
                }
    """

    def thread_process(tokenizer1, start, end,
                       out_list_inputs_text, out_list_labels_text,
                       out_list_input_ids, out_list_attention_mask, out_lst_labels):
        if start == 0:
            t = tqdm(range(start, end))
        else:
            t = range(start, end)
        for i in t:
            sample = samples[i]
            ques = sample['instruction'] + sample['input']
            ans = sample['output']

            out_list_inputs_text[i] = ques
            out_list_labels_text[i] = ans
            tokenized_ques = tokenizer1(ques, max_length=args['max_length_input'],
                                        truncation=True, add_special_tokens=True)
            tokenized_ans = tokenizer1(ans, max_length=args['max_length_output'],
                                       truncation=True, add_special_tokens=True)
            out_list_input_ids[i] = tokenized_ques['input_ids'].copy()
            out_list_attention_mask[i] = tokenized_ques['attention_mask'].copy()
            out_lst_labels[i] = tokenized_ans['input_ids'].copy()
        # print(f"{start}~{end} tokenized")

    len_samples = len(samples)
    print(f"length of dataset is {len_samples}")
    outputs = {'inputs_text': [''] * len_samples,
               'labels_text': [''] * len_samples,
               'input_ids': [[]] * len_samples,
               'attention_mask': [[]] * len_samples,
               'labels': [[]] * len_samples}

    thread_num = 8
    thread_start_list = [len_samples * i // thread_num for i in range(thread_num)]
    thread_start_list.append(len_samples)
    thread_list = []
    for i in range(thread_num):
        thread_ = threading.Thread(target=thread_process,
                                   args=(tokenizer,
                                         thread_start_list[i],
                                         thread_start_list[i + 1],
                                         outputs['inputs_text'],
                                         outputs['labels_text'],
                                         outputs['input_ids'],
                                         outputs['attention_mask'],
                                         outputs['labels'],
                                         ))
        thread_list.append(thread_)
        thread_list[i].start()
    for i in range(thread_num):
        thread_list[i].join()  # wait until thread finished

    # for sample in tqdm(samples):
    #     outputs['inputs_text'].append(sample['ques'])
    #     outputs['labels_text'].append(sample['ans'])
    #     tokenized_ques = tokenizer(sample['ques'], max_length=512, truncation=True)
    #     tokenized_ans = tokenizer(sample['ans'], max_length=512, truncation=True)
    #     outputs['input_ids'].append(tokenized_ques['input_ids'].copy())
    #     outputs['attention_mask'].append(tokenized_ques['attention_mask'].copy())
    #     outputs['labels'].append(tokenized_ans['input_ids'].copy())

    # tokenized_questions = tokenizer(lst_ques, max_length=512, truncation=True)
    # tokenized_answers = tokenizer(lst_ans, max_length=512, truncation=True)
    # ##### {'input_ids': [[], [], ...], 'attention_mask': [[], [], ...]}
    #
    # outputs['inputs_text'] = lst_ques
    # outputs['labels_text'] = lst_ans
    # outputs['input_ids'] = tokenized_questions['input_ids']
    # outputs['attention_mask'] = tokenized_questions['attention_mask']
    # outputs['labels'] = tokenized_answers['input_ids']
    outputs = Dataset.from_dict(outputs)

    # contents = ["摘要生成: \n" + e for e in exmaples["content"]]
    # inputs = tokenizer(contents, max_length=384, truncation=True)
    # labels = tokenizer(text_target=exmaples["title"], max_length=64, truncation=True)
    # inputs["labels"] = labels["input_ids"]
    # return inputs
    return outputs


def tokenizer_ids_to_str(tokenizer, ids, args: dict):

    text = ''
    if args['tokenizer_ids_to_str'] in ['Randeng-T5-77M-MultiTask-Chinese']:
        text = tokenizer.decode(ids)
        # find </s>, <pad>
        text = text.replace('</s>', '')
        text = text.replace('<pad>', '')
        # find diy special tokens
        for item in args['special_tokens']:
            text = text.replace(f' {item}', f'{item}')
            text = text.replace(f'{item} ', f'{item}')
        # tokens = tokenizer.convert_ids_to_tokens(ids)
        # for i in range(len(tokens)):
        #     # find '▁'
        #     if tokens[i][0] == '▁':
        #         if i == 0:
        #             tokens[i] = tokens[i][1:]
        #         else:
        #             if tokens[i-1] in list(tokenizer.added_tokens_encoder.keys()):
        #                 tokens[i] = tokens[i][1:]
        #             else:
        #                 if len(tokens[i]) == 1:
        #                     tokens[i] = ' '
        #                 else:
        #                     tokens[i] = tokens[i][1:]
        #     # find </s>, <pad>
        #     if tokens[i] in ['</s>', '<pad>']:
        #         tokens[i] = ''
        #     # find <unk>
        #     if tokens[i] in ['<unk>', ]:
        #         tokens[i] = '<u>'
        # text = ''.join(tokens)

    return text


def dataset_predict(trainer, tokenizer, dataset, args):
    """
    用模型推理一个数据集文件中所有样本，返回问题、标签、预测答案。
    :param trainer:
    :param tokenizer:
    :param dataset:
        dataset[i] = {
            'input_ids': [64790, 64792, 64795, 30910, 13, 30910, 40369, 55390, 54538, 31775, 54541, 30989, 31877, 31201, 33134, 32195, 31201, 37421, 54532, 31201, 32083, 31201, 31731, 31201, 33328, 31201, 37811, 31201, 37027, 54536, 31201, 34037, 31201, 34579, 31201, 54835, 54653, 31201, 48763, 54541, 31201, 32633, 30991, 54530, 35414, 54570, 31123, 55351, 31631, 55066, 33287, 54716, 31155, 31002, 39099, 30994, 6827, 283, 30964, 30959, 54542, 8473, 16696, 30959, 54620, 55037, 54532, 5321, 30950, 30941, 30964, 30959, 31155, 64796],
            'output_ids': [30910, 13, 906, 33227, 30994, 5321, 30950, 30941, 30964, 30959, 31002, 31775, 30994, 37811, 31002, 54992, 54618, 30994, 6827, 283, 30964, 30959, 31002, 54992, 54618, 30994, 8473, 16696, 30959, 2]
        }
    :return:
        预计得到的文件格式：
            [{'ques': 'sent', 'label_ans': '...', 'pred_ans': '...'}, {}, ...]
    """

    res = trainer.predict(dataset)  # !!!!!
    # print(res)
    # print(xxxxx)
    """ res = 
        PredictionOutput(
            predictions=array(
                [
                    [ 30910,    13,   906, 33227, 30994,  6827,   283, 30964, 30959, ... ],
                    [ 30910,    13,   906, 33227, 30994, 30952, 30929, 30947,  5565, ... ]
                ]
            ), 
            label_ids=array(
                [
                    [ 30910,    13,   906, 33227, 30994,  5321, 30950, 30941, 30964, ... ],
                    [ 30910,    13,   906, 33227, 30994, 30952, 30929, 30947,  5565, ... ]
                ]
            ), 
            metrics={'test_rouge-1': 88.46155, 'test_rouge-2': 78.14815, 'test_rouge-l': 70.8498, 'test_bleu-4': 0.5293067243030543, 'test_runtime': 8.5984, 'test_samples_per_second': 0.233, 'test_steps_per_second': 0.116}
        )
    """
    assert len(res.predictions) == len(dataset)

    data_out = []
    for i in range(len(res.predictions)):
        # print(f"\n{i}")
        ques_ids = dataset[i]['input_ids']
        # label_ans1_ids = dataset[i]['output_ids']
        label_ids_from_dataset = dataset[i]['labels']
        pred_ans_ids = res.predictions[i].tolist()
        label_ids_from_trainer_pred = res.label_ids[i].tolist()  # padded with 0, longer
        assert label_ids_from_dataset == label_ids_from_trainer_pred[:len(label_ids_from_dataset)], \
            f"\n{ques_ids}\n{label_ids_from_dataset}\n{label_ids_from_trainer_pred}"

        # print("tokenizer.tokenizer.index_special_tokens")
        # print(tokenizer.tokenizer.index_special_tokens)

        # ques_tokens = tokenizer.convert_ids_to_tokens(ques_ids)
        # ques_str = ''.join(ques_tokens)
        # print(f"-- ques_tokens: {ques_tokens}")
        # print(f"-- ques_str: {ques_str}")

        ques_str = tokenizer_ids_to_str(tokenizer, ques_ids, args=args)
        pred_ans_str = tokenizer_ids_to_str(tokenizer, pred_ans_ids, args=args)
        label_ans_str = tokenizer_ids_to_str(tokenizer, label_ids_from_dataset, args=args)

        token_i = len(pred_ans_ids)
        while pred_ans_ids[token_i-1] == 0:
            token_i -= 1
        pred_ans_ids_no_pad = pred_ans_ids[:token_i]
        data_out.append({
            'ques': ques_str,
            'label_ans': label_ans_str,
            'pred_ans': pred_ans_str,
        })
        # 'pred_ans_ids': pred_ans_ids_no_pad

    return data_out


def inference(trainer, tokenizer, dataset_dict, args):
    """

    :param trainer:
    :param tokenizer:
    :param args:  yaml 文件配置参数
    :return:
    """

    # 读取路径下所有checkpoint路径
    checkpoint_dir_list = []
    folders = list(os.walk(args['output_dir']))[0][1]
    for folder in folders:
        if 'checkpoint' in folder:
            checkpoint_dir_list.append(os.path.join(args['output_dir'], folder))
    checkpoint_dir_list.sort(key=lambda x: int(suffix_find(x, 'point-')), reverse=True)
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(10)

    # 进入各checkpoint下，读取参数进行验证
    for checkpoint_dir in checkpoint_dir_list:
        trainer._load_from_checkpoint(resume_from_checkpoint=checkpoint_dir)

        # 训练集抽样预测
        output_file = os.path.join(checkpoint_dir, "dataset_prediction_train.json")
        if dataset_dict['train'] is not None and os.path.isfile(output_file) is False:
            print(f"\n-- ({checkpoint_dir}) train dataset prediction ing...")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['train'].select(list(range(2))), args)
            print(f"train example: {data_out[0]}")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['train'].select(list(range(200))), args)
            with open(output_file, "w", encoding="utf-8") as fp:
                json.dump(data_out, fp, ensure_ascii=False, indent=4)

        # 验证集预测
        output_file = os.path.join(checkpoint_dir, "dataset_prediction_dev.json")
        if dataset_dict['dev'] is not None and os.path.isfile(output_file) is False:
            print(f"\n-- ({checkpoint_dir}) dev dataset prediction ing...")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['dev'].select(list(range(10))), args)
            print(f"dev example: {data_out[0]}")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['dev'], args)
            with open(output_file, "w", encoding="utf-8") as fp:
                json.dump(data_out, fp, ensure_ascii=False, indent=4)

        # 测试集预测
        output_file = os.path.join(checkpoint_dir, "dataset_prediction_test.json")
        if dataset_dict['test'] is not None and os.path.isfile(output_file) is False:
            print(f"\n-- ({checkpoint_dir}) test dataset prediction ing...")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['test'].select(list(range(10))), args)
            print(f"dev example: {data_out[0]}")
            data_out = dataset_predict(trainer, tokenizer, dataset_dict['test'], args)
            with open(output_file, "w", encoding="utf-8") as fp:
                json.dump(data_out, fp, ensure_ascii=False, indent=4)


def main():
    # ---------------------------------------- parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./config/240921.yaml",
                        help="./config/240527.yaml")  # 程序运行配置文件，不是预训练模型读取的config
    parser.add_argument('--train_or_infer', type=str, default="train",
                        help="<train_infer> <train> <infer>")  # 训练还是预测，也可以同时
    args_from_parser = parser.parse_args()

    # ---------------------------------------- yaml config
    with open(args_from_parser.config_file, 'r', encoding='utf-8') as file:
        args_from_yaml = yaml.safe_load(file)   # a dict
    print(f"yaml config =")
    for item in args_from_yaml.items():
        print(f"  {item}")
        # time.sleep(1)
    print("")

    # ---------------------------------------- pretrain model config & tokenizer
    config = AutoConfig.from_pretrained(args_from_yaml['model_name_or_path'])
    print(f"pretain model config = {config}\n")

    tokenizer = T5Tokenizer.from_pretrained(args_from_yaml['model_name_or_path'], use_fast=False)
    special_tokens_list = args_from_yaml['special_tokens']
    if len(special_tokens_list) > 0:
        tokenizer.add_tokens(special_tokens_list)
        # tokenizer.add_tokens(special_tokens_list)

    # ---------------------------------------- data load
    print("-- dev_dataset loading ...")
    with open(args_from_yaml['dev_data_path'], "r", encoding="utf-8") as f1:
        data = json.loads(f1.read())
    dev_dataset = dataset_load(data[:], tokenizer, args_from_yaml)
    print("-- test_dataset loading ...")
    with open(args_from_yaml['test_data_path'], "r", encoding="utf-8") as f1:
        data = json.loads(f1.read())
    test_dataset = dataset_load(data[:], tokenizer, args_from_yaml)
    print("-- train_dataset loading ...")
    with open(args_from_yaml['train_data_path'], "r", encoding="utf-8") as f1:
        data = json.loads(f1.read())
    if 'train' in args_from_parser.train_or_infer:
        train_dataset = dataset_load(data[:], tokenizer, args_from_yaml)
    else:
        train_dataset = dataset_load(data[:1000], tokenizer, args_from_yaml)
    del data
    # example:
    print("example:")
    print(f"input_text = {dev_dataset['inputs_text'][0]}")
    print(f"label_text = {dev_dataset['labels_text'][0]}")
    print(f"input_ids = {dev_dataset['input_ids'][0]}")
    print(f"label_ids = {dev_dataset['labels'][0]}")
    print(f"input_tokens = \n{tokenizer.convert_ids_to_tokens(dev_dataset['input_ids'][0])}")
    print(f"label_tokens = \n{tokenizer.convert_ids_to_tokens(dev_dataset['labels'][0])}")
    print(f"input_decode_tokens = \n{[tokenizer.decode(id_) for id_ in dev_dataset['input_ids'][0]]}")
    print(f"label_decode_tokens = \n{[tokenizer.decode(id_) for id_ in dev_dataset['labels'][0]]}")
    print(f"input_decode = \n{tokenizer.decode(dev_dataset['input_ids'][0])}")
    print(f"label_decode = \n{tokenizer.decode(dev_dataset['labels'][0])}")
    print(f"my input_decode = \n"
          f"{tokenizer_ids_to_str(tokenizer, dev_dataset['input_ids'][0], args=args_from_yaml)}")
    print(f"my label_decode = \n"
          f"{tokenizer_ids_to_str(tokenizer, dev_dataset['labels'][0], args=args_from_yaml)}")
    print("")
    # print(tokenizer.convert_ids_to_tokens([1903, 2,1904, 32597, 1, 0, 0]))  # ['无', '<结束>', '</s>', '<pad>', '<pad>']
    # print(tokenizer.decode([1903, 2, 1904, 32597, 1, 0, 0]))  # 无<unk>button <结束> </s><pad><pad>
    # print(tokenizer.convert_tokens_to_ids(['<unk>']))
    # print(list(tokenizer.added_tokens_encoder.keys()))
    ques_tok_len = Histogram(300, 1300, 100)
    for i, item in enumerate(train_dataset['input_ids']):
        if i % 36 < 3:
            ques_tok_len.input_one_data(len(item))
    ques_tok_len.update_ratio()
    print(ques_tok_len.statistic_info_simple, ques_tok_len.larger_num)
    ans_tok_len = Histogram(300, 1300, 100)
    for i, item in enumerate(train_dataset['labels']):
        if i % 36 < 3:
            ans_tok_len.input_one_data(len(item))
    ans_tok_len.update_ratio()
    print(ans_tok_len.statistic_info_simple, ans_tok_len.larger_num)
    # print(xxxxx)
    time.sleep(10)

    # ---------------------------------------- pretrained model
    # model = AutoModelForSeq2SeqLM.from_pretrained("./models/Randeng-T5-77M-MultiTask-Chinese")
    model = T5ForConditionalGeneration.from_pretrained(args_from_yaml['model_name_or_path'])
    if len(special_tokens_list) > 0:
        print("")
        print(f"vocab_size before add special token: {tokenizer.vocab_size}")
        model.resize_token_embeddings(tokenizer.vocab_size + len(tokenizer.added_tokens_encoder))
        print(f"vocab_size before add special token: {tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)}")
        # print(xxxxx)

    # ---------------------------------------- build transformers trainer
    train_batch_size = args_from_yaml['per_device_train_batch_size']
    train_samples_num = len(train_dataset)
    checkpoint_num = args_from_yaml['checkpoint_num']
    max_epoch = args_from_yaml['max_epoch']
    args = Seq2SeqTrainingArguments(
        output_dir=args_from_yaml['output_dir'],
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=args_from_yaml['per_device_eval_batch_size'],
        gradient_accumulation_steps=args_from_yaml['gradient_accumulation_steps']//train_batch_size,

        max_steps=int(train_samples_num*max_epoch) // train_batch_size,
        evaluation_strategy="steps",
        eval_steps=int(train_samples_num*max_epoch) // (checkpoint_num*train_batch_size),
        save_strategy="steps",
        save_steps=int(train_samples_num*max_epoch) // (checkpoint_num*train_batch_size),
        logging_steps=args_from_yaml['logging_steps'],

        metric_for_best_model="rouge-2",
        predict_with_generate=True,

        fp16=args_from_yaml['fp16'],
        warmup_ratio=0.1,
        learning_rate=float(args_from_yaml['learning_rate']),
        max_grad_norm=0.2,     # 梯度裁剪
    )
    print(f"train_samples_num = {train_samples_num}\nsave_steps = {args.save_steps}")
    time.sleep(5)

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset.select(list(range(200))),
        compute_metrics=functools.partial(compute_metric, tokenizer=tokenizer),   # eval func
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding='longest',)
    )

    # ---------------------------------------- train
    if 'train' in args_from_parser.train_or_infer:
        trainer.train()

        # 删除无用的大文件
        command = f"find {args_from_yaml['output_dir']} -type f -name 'optimizer.pt' | xargs rm -f"
        subprocess.run(command, shell=True, capture_output=True, text=True)

    # ---------------------------------------- infer dev & test
    if 'infer' in args_from_parser.train_or_infer:
        inference(trainer, tokenizer,
                  dataset_dict={'train': train_dataset,
                                'dev': dev_dataset,
                                'test': test_dataset},
                  args=args_from_yaml)



if __name__ == '__main__':
    main()

"""
    parser.add_argument('--config_file', type=str, default="./config/240530.yaml",
                        help="./config/240527.yaml")  # 程序运行配置文件，不是预训练模型读取的config
    parser.add_argument('--train_or_infer', type=str, default="train",
                        help="<train_infer> <train> <infer>")  # 训练还是预测，也可以同时

python finetune240517.py --config_file config/240921.yaml --train_or_infer train_infer

实验调整：
-、修改max_length，程序能否正常运行？
-、不同的T5模型，配置上有什么不同？
"""
