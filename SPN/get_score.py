"""
更新说明：除了从label、pred中提取主客体那一部分。其他部分发生任何改动，version更新。

version: 240917
    -- 修改 evaluate_1_checkpoint__for_SPN
version: 240901
    -- 写了 evaluate_1_checkpoint__for_TPLinker , 添加到 evaluate_1_checkpoint 选项中。
version: 240826
    -- 写了 evaluate_1_checkpoint__for_BiRTE , 添加到 evaluate_1_checkpoint 选项中。
    -- 减少了对本地其他文件的依赖
version: 240721
    -- 整合了 our model 中的 evaluate_1_checkpoint，将 evaluate_1_checkpoint 改为一个函数指针。
        因此目前该程序能直接用于 our model 与 SPN ，只需修改 evaluate_1_checkpoint
version: 240711
    -- 将大部分 配置参数 改写为可通过命令行输入的格式（argparse）
"""

import os
import time
import json
import yaml
import argparse
import matplotlib.pyplot as plt

import jieba
import rouge_chinese

""" ours
python get_score.py --OUTPUT_DIR="outputs/240903 Dataset Chg/240903_LSTM-nBiL1H768_LossLSR" --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6

"""

""" BiRTE
python3 get_score.py \
    --MODEL_NAME="BiRTE" \
    --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
    --DATASET_DIR="datasets/CMIM2023-NOM-task1-Re" \
    --LABEL_FILENAME_dev="dev.json" --LABEL_FILENAME_test="test.json"\
    --OUTPUT_DIR="outputs/240825_SeqLen200_lr3e-5/" \
    --PREDICT_FILENAME_dev="dev_pred.json" --PREDICT_FILENAME_test="test_pred.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
    
"""

""" TPLinker
python3 get_score.py \
    --MODEL_NAME="tplinker" \
    --PRETRAIN_MODEL_DIR="models/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM2023-KG-task1-RRA/240607_seed0" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="tplinker/default_log_dir/240607" \
    --PREDICT_FILENAME_dev="prediction_valid.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

存在一些额外参数，在 evaluate_1_checkpoint__for_TPLinker 函数开头。
"""

""" SPN
python3 get_score.py \
    --MODEL_NAME="SPN" \
    --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM2023-KG-task1-RRA/240607_seed0" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="tplinker/default_log_dir/240607" \
    --PREDICT_FILENAME_dev="prediction_valid.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

"""

parser = argparse.ArgumentParser()

# ---------- 任务
parser.add_argument('--MODEL_NAME', type=str, default="ours",
                    choices=['ours', 'SPN', 'BiRTE', 'tplinker'])
parser.add_argument('--SPECIAL_TASK', type=str, default="none",
                    choices=['none', 'segmented_entity'])
# ^^^ SPECIAL_TASK 特殊任务
#      none：对比所有样本所有三元组，输出 p，r，f1
#      segmented_entity：对比所有样本中，含分段实体的三元组，输出 p，r，f1；不适合rouge

# ---------- 预测文件
parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/LSTM compare in LossCE/240725_LSTM-BiL2H576_LossCE")
# parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/nyt/nyt_LSTM-BiL2H576_LossCE")
parser.add_argument('--CHECKPOINT_FOLDER_PREFIX', type=str, default="checkpoint-epoch")  # 各checkpoint文件夹除数字部分外的前缀
parser.add_argument('--PREDICT_FILENAME_dev', type=str, default="predict_triples_dev.txt")
parser.add_argument('--PREDICT_FILENAME_test', type=str, default="predict_triples_test.txt")

# ---------- 标签文件
parser.add_argument('--DATASET_DIR', type=str, default="dataset/CMIM2023-KG-task1-RRA/groups/240607_seed0_json_desensitize_1")
# parser.add_argument('--DATASET_DIR', type=str, default="dataset/nyt")
parser.add_argument('--LABEL_FILENAME_dev', type=str, default="valid_data.json")
parser.add_argument('--LABEL_FILENAME_test', type=str, default="test_data.json")

parser.add_argument('--PRETRAIN_MODEL_DIR', type=str, default="lib/prev_trained_model/bert-base")
# parser.add_argument('--PRETRAIN_MODEL_DIR', type=str, default="lib/prev_trained_model/bert-base-cased")

# rouge
parser.add_argument('--USE_ROUGE', type=bool, default=False)
parser.add_argument('--WHICH_ROUGE', type=str, default="rouge-1")
parser.add_argument('--ROUGE_THRE', type=float, default=0.5)
parser.add_argument('--TOKEN_TYPE', type=str, default='tokenizer',
                    choices=['jieba', 'tokenizer'])

# delete bad model
parser.add_argument('--DELETE_BAD_MODEL', type=bool, default=False)  # 功能开关
parser.add_argument('--CHECKPOINT_MODEL_NAME', type=str, default="pytorch_model.bin")
parser.add_argument('--BAD_MODEL_THRE', type=float, default=0.5)  # 效果差的模型的f1阈值

args_global = parser.parse_args()


# # tokenizer
# from transformers import BertTokenizer
# # from dataset_loader import ADDITIONAL_SPECIAL_TOKENS
# tokenizer_global = BertTokenizer.from_pretrained(args_global.PRETRAIN_MODEL_DIR, do_lower_case=True)
# # tokenizer_global = BertTokenizerFast.from_pretrained(
# #         PRETRAIN_MODEL_DIR, do_basic_tokenize=False,
# #         add_special_tokens=True, do_lower_case=True
# #     )
# ADDITIONAL_SPECIAL_TOKENS = ['“', '”']
# tokenizer_global.add_tokens(ADDITIONAL_SPECIAL_TOKENS)

# Relation_List = ['含有', '手段采用', '属性有', '组成部分', '功能', '实例为', '前提是', '特点', '定义', '影响', '别名', '造成', '分类']
# Definition, Alias, Instance, Characteristic,
# Contains, Attribute, Component, Category,
# Function, Approach, Premise, Cause, Influence
# 定义, 别名, 实例, 特点,    含有, 有属性, 组成部分, 分类,   功能, 实现手段, 前提, 造成, 影响
# [Contains', 'means used', 'attributes have', 'components',' functions', 'instances are', 'premises are', 'characteristics',' definitions', 'impacts',' aliases', 'causes',' classifications']

def process_after_BertTokenizer_decode(text_decode):  # jyz add 2024-07
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        run_args.model_dir, additional_special_tokens=added_token, do_basic_tokenize=False,
        add_special_tokens=True, do_lower_case=True)
    以上 tokenizer decode 时，汉字之间都会有空格，开头的##不会消除，因此进行手动处理
    """
    a2z = "abcdefghijklmnopqrstuvwxyz"
    text = ""
    # 手动去除token间的空格，但保留英文单词间的
    for i in range(len(text_decode)):
        if text_decode[i] == " ":
            if text_decode[i - 1] in a2z and text_decode[i + 1] in a2z:
                text += text_decode[i]
        else:
            text += text_decode[i]
    # 去除首尾的特殊字符
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.strip("#")
    # 将[UNK]改为一个字符
    text = text.replace("[UNK]", "?")
    return text


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
        sub_pos_l += symbol_l_pos2 + len(symbol_l_str)
        sub_str = sub_str[symbol_l_pos2 + len(symbol_l_str):]
        symbol_l_pos2 = sub_str.find(symbol_l_str)

    return sub_str


def str_left_add(str_in: str, char: str, max_len: int):
    """
    在字符串左侧补 char，直到到达最大长度
    """
    while len(str_in) < max_len:
        str_in = char + str_in
    return str_in


def span_find(input_str, span_l_str, span_r_str, start=0):
    """
    find the next sub-string between "span_l_str" and "span_r_str" in "input_str"

    :param input_str:
    :param span_l_str: left boundary symbol of span
    :param span_r_str: right boundary symbol of span
    :param start: starting position for search
    :return: (sub_string, sub_string_left_position, sub_string_right_positon)

    example:
        1. span_find("abc[123]defg[45]hijk", "[", "]", 0)
           return is ('123', 4, 7)
        2. span_find("abc[123]defg[45]hijk", "[", "]", 7)
           return is ('45', 13, 15)
        3. span_find("abc[123]defg[45]hijk", "[", "]", 15)
           return is ('', -1, -1)
        4. span_find("abc[123]defg[45]hijk", "[", "]", 13)
           return is ('', -1, -1)
    """

    span_l_pos = input_str.find(span_l_str, start)  # find the position of left boundary symbol of span
    if span_l_pos < 0:
        return "", -1, -1
    sub_l_pos = span_l_pos + len(span_l_str)

    span_r_pos = input_str.find(span_r_str, span_l_pos)  # find the position of right boundary symbol of span
    if span_r_pos < 0:
        return "", -1, -1
    sub_r_pos = span_r_pos

    sub_str = input_str[sub_l_pos:sub_r_pos]
    return sub_str, sub_l_pos, sub_r_pos


def my_text_cut(text, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))
    # for i in range(len(tokens) - 1, -1, -1):
    #     if tokens[i][0:2] == '##' and i > 0:
    #         tokens[i - 1] = tokens[i - 1] + tokens[i][2:]
    #         del tokens[i]
    return tokens  # a list


def span_have_overlap(span1, span2):
    # 判断是否有重叠
    # 情况1: span1的结束值大于span2的起始值，且span1的起始值小于span2的结束值
    # 情况2: span2的结束值大于span1的起始值，且span2的起始值小于span1的结束值
    # 两种情况之一为真，则存在重叠

    # 提取跨度的起始和结束值
    x1, y1 = span1
    x2, y2 = span2
    return (x1 < y2 and x2 < y1) or (x2 < y1 and x1 < y2)


def get_rouge(pred_txt: str, label_txt: str,
              args, tokenizer):  # 获取rouge分数
    if args.TOKEN_TYPE == 'jieba':
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
    elif args.TOKEN_TYPE == 'tokenizer':
        pred_tokens = my_text_cut(pred_txt, tokenizer)
        label_tokens = my_text_cut(label_txt, tokenizer)
    if len(pred_tokens) == 0:  # 为了防止 pred_tokens 为空
        pred_tokens = ['-+-+']
    assert len(label_tokens) > 0, f"\n{pred_txt}\n{label_txt}"

    rouge = rouge_chinese.Rouge()
    scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
    """scores = 
    [{
        'rouge-1': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}, 
        'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
        'rouge-l': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}
    }]
    example: 
        [a], [a b] --> rouge1: r=0.5, p=1, f1=0.67
        [a], [a b c] --> rouge1: r=0.33, p=1, f1=0.5(0.499...)
        [a b], [a c] --> rouge1: r=0.5, p=0.5, f1=0.5(0.499...)
    """
    return scores


def get_rouge_test(pred_txt: str, label_txt: str, tokenizer):  # 获取rouge分数
    pred_tokens = my_text_cut(pred_txt, tokenizer)
    print(f"pred_tokens = {pred_tokens}")
    label_tokens = my_text_cut(label_txt, tokenizer)
    print(f"label_tokens = {label_tokens}")
    if len(pred_tokens) == 0:  # 为了防止 pred_tokens 为空
        pred_tokens = ['-+-+']
    assert len(label_tokens) > 0, f"\n{pred_txt}\n{label_txt}"

    rouge = rouge_chinese.Rouge()
    scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
    print(scores)
    """scores = 
    [{
        'rouge-1': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}, 
        'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
        'rouge-l': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}
    }]
    example: 
        [a], [a b] --> rouge1: r=0.5, p=1, f1=0.67
        [a], [a b c] --> rouge1: r=0.33, p=1, f1=0.5(0.499...)
        [a b], [a c] --> rouge1: r=0.5, p=0.5, f1=0.5(0.499...)
    """
    return scores


def get_best_rouge_in_labels(triple_pred, triples_label_remain: list, args, tokenizer):
    """
        在考虑 rouge 的前提下，判断 triple_pred 是否存在与 triples_label_remain 中。
        1、在triples_label_remain中，查找与triple_pred最相似的triple_label
            最相似的标准为：关系必须相同，主体客体rouge中的较小值最大
        2、若最相似的rouge大于阈值，认为triple_pred存在于triples_label_remain
        3、若triple_pred存在于triples_label_remain，返回True，并删除triples_label_remain
            中的相应三元组并返回；若不存在，返回False，并返回triples_label_remain
    """
    assert len(triple_pred) == 3, f"\n{triple_pred}"

    best_label = {"triple_i": None, "rouge_score": 0}
    for triple_i in range(len(triples_label_remain)):
        triple_label = triples_label_remain[triple_i]
        # relation not same
        if triple_pred['rela'] != triple_label['rela']:
            continue
        # 获取 pred、label 主体间的 rouge
        subj_rouge_score = get_rouge(triple_pred['subj'], triple_label['subj'],
                                     args=args, tokenizer=tokenizer)[0]
        # 客体间的 rouge
        obj_rouge_score = get_rouge(triple_pred['obj'], triple_label['obj'],
                                    args=args, tokenizer=tokenizer)[0]
        # 融合rouge分数的策略1: 主客体 WHICH_ROUGE 的较小值
        triple_score = min(subj_rouge_score[args.WHICH_ROUGE]['f'],
                           obj_rouge_score[args.WHICH_ROUGE]['f'])
        # 更新best
        if triple_score > best_label['rouge_score']:
            best_label['rouge_score'] = triple_score
            best_label['triple_i'] = triple_i
            if best_label['rouge_score'] > 0.99:  # == 1.0
                break

    return best_label
    # if best_label['rouge_score'] > args.ROUGE_THRE:   # 阈值
    #     del triples_label_remain[best_label['triple_i']]
    #     return True
    # else:
    #     return False


def f1_score_triple(preds: list, labels: list, args, tokenizer):
    """
    if 1 triple in preds is also in labels as well, correct + 1
    :param preds: [
                    [triple1_1, triple1_2, ...],
                    [triple2_1, triple2_2, ...],
                        ...
                  ]
    :param labels: same as preds
    :return:
    """

    assert len(labels) == len(preds)
    assert type(labels[0]) == list
    assert type(preds[0]) == list, f"\n{preds[0]}"

    correct_have, guess_have, gold_have = 0, 0, 0
    # 有关且预测正确，       总预测有关，            实际有关
    gold_no = 0  # 实际非该标签
    # guess_is_lbl_when_gold_is_lbl = 0  # 实际为该标签的样本中，预测为该标签
    guess_have_when_gold_no = 0  # 实际非该标签的样本中，预测为该标签

    for i in range(len(preds)):

        # 获取每个样本的三元组
        triples_pred = preds[i]
        triples_label = labels[i]  # [(s, r, o), ...]

        guess_have += len(triples_pred)
        gold_have += len(triples_label)

        if args.USE_ROUGE is False:
            # 普通判别是否正确
            for triple_pred in triples_pred:
                assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
                if triple_pred in triples_label:
                    correct_have += 1
        else:
            # 添加rouge，判别是否正确
            triples_label_remain = triples_label.copy()
            for triple_pred in triples_pred:
                assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
                len_triples_label_remain = len(triples_label_remain)
                best_label = get_best_rouge_in_labels(triple_pred, triples_label_remain,
                                                      args=args, tokenizer=tokenizer)
                # if rouge_triple_in_labels(triple_pred, triples_label_remain, args=args):
                if best_label['rouge_score'] > args.ROUGE_THRE:
                    del triples_label_remain[best_label['triple_i']]
                    # assert len(triples_label_remain) == len_triples_label_remain - 1
                    correct_have += 1

    p_micro = 1.0
    if guess_have > 0:
        p_micro = float(correct_have) / float(guess_have)
    r_micro = 0.0
    if gold_have > 0:
        r_micro = float(correct_have) / float(gold_have)
    f1_micro = 0.0
    if p_micro + r_micro > 0.0:
        f1_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro)
    # print('  {}/{}, {}/others, {}/{}/{}, "{}"'.format(
    #     correct_have, gold_have, guess_have_when_gold_no,
    #     str(p_micro)[:4], str(r_micro)[:4], str(f1_micro)[:4], which_label))
    return p_micro, r_micro, f1_micro


def f1_score_sample(preds, labels):
    """
    if 1 triple in preds is also in labels as well, correct + 1
    :param preds: [
                    [triple1_1, triple1_2, ...],
                    [triple2_1, triple2_2, ...],
                        ...
                  ]
    :param labels: same as preds
    :return:
    """
    assert len(labels) == len(preds)

    correct_have, guess_have, gold_have = 0, 0, 0
    # 有关且预测正确，       总预测有关，            实际有关
    gold_no = 0  # 实际非该标签
    # guess_is_lbl_when_gold_is_lbl = 0  # 实际为该标签的样本中，预测为该标签
    guess_have_when_gold_no = 0  # 实际非该标签的样本中，预测为该标签

    for i in range(len(preds)):

        triples_pred = preds[i]
        triples_label = labels[i]  # [(s, r, o), ...]

        if len(triples_pred) > 0:
            guess_have += 1
        if len(triples_label) > 0:
            gold_have += 1

        correct = 1
        for triple_pred in triples_pred:
            if triple_pred not in triples_label:
                correct = 0
                break
        for triple_label in triples_label:
            if triple_label not in triples_pred:
                correct = 0
                break
        correct_have += correct

    p_micro = 1.0
    if guess_have > 0:
        p_micro = float(correct_have) / float(guess_have)
    r_micro = 0.0
    if gold_have > 0:
        r_micro = float(correct_have) / float(gold_have)
    f1_micro = 0.0
    if p_micro + r_micro > 0.0:
        f1_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro)
    # print('  {}/{}, {}/others, {}/{}/{}, "{}"'.format(
    #     correct_have, gold_have, guess_have_when_gold_no,
    #     str(p_micro)[:4], str(r_micro)[:4], str(f1_micro)[:4], which_label))
    return p_micro, r_micro, f1_micro


def evaluate_1_checkpoint__for_ourmodel(predict_file, label_file, tokenizer):
    from dataset_loader import relation_modify

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = eval(f.read())
        """
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(len(predict_data), len(label_data))
    assert len(predict_data) == len(label_data)

    all_samples_triples_label = []
    all_samples_triples_pred = []
    pred_data_i = 0
    for data_i in range(len(label_data)):  # 遍历标签

        # -------------------- process label
        # triples_label_with_pos = label_data[data_i][2].copy()
        # ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        # triples_label = []
        # for triple_str_pos in triples_label_with_pos:
        #     triple_str = tuple(triple_str_pos[0])
        #     # subj_str = token_decode(tokenizer_global, tokenizer_global.encode(subj_str_origin))
        #     # obj_str = token_decode(tokenizer_global, tokenizer_global.encode(obj_str_origin))
        #     if triple_str not in triples_label:
        #         triples_label.append(triple_str)
        triples_label_with_pos = label_data[data_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            if args_global.SPECIAL_TASK in ["segmented_entity"]:
                if len(subj_char_span) == 1 and len(obj_char_span) == 1:  # 不含分段实体
                    continue

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                # triple_info['subj'] = subj_str,   # 实体取最后一个单词
                # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                triple_info['rela'] = relation_modify(rela_str, mode='webnlg')
                triple_info['obj'] = obj_str.split()[-1].lower()
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_with_pos = predict_data[data_i][1].copy()
        ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        triples_pred = []
        for triple_str_pos in triples_pred_with_pos:
            subj_str = triple_str_pos[0][0]
            rela_str = triple_str_pos[0][1]
            obj_str = triple_str_pos[0][2]
            subj_char_span = triple_str_pos[1]
            obj_char_span = triple_str_pos[2]

            if args_global.SPECIAL_TASK in ["segmented_entity"]:
                if len(subj_char_span) == 1 and len(obj_char_span) == 1:  # 不含分段实体
                    continue

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                # triple_info['subj'] = subj_str,
                # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                # triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                triple_info['rela'] = relation_modify(rela_str, mode='webnlg')
                triple_info['obj'] = obj_str.split()[-1].lower()
            else:
                triple_info['subj'] = subj_str.replace('爱立信', '安尔信')  # 兼容操作
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str.replace('爱立信', '安尔信')
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)

        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f"triples_pred{data_i} = {triples_pred}")

        # triples_label_str = [tuple(triple[0]) for triple in triples_label]
        # triples_pred_str = [(triple['subject'], triple['predicate'], triple['object']) for triple in triples_pred_gather]
        # triples_pred_str = list(set(triples_pred_str))
        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    # input format:
    # print(all_samples_triples_pred[10])
    # print(all_samples_triples_label[10])
    print(f"triple_label_num={sum([len(triples) for triples in all_samples_triples_label])}, "
          f"triple_pred_num={sum([len(triples) for triples in all_samples_triples_pred])}")
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple


def evaluate_1_checkpoint__for_BiRTE(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """
            {
                "text": "系统消息4包含LAI、RACH控制参数信息。",
                "triple_list_pred": [
                    [ "系统消息4", "含有", "LAI控制参数信息" ],
                    [ "系统消息4", "含有", "RACH控制参数信息" ]
                ],
            },
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(len(predict_data), len(label_data))
    assert len(predict_data) == len(label_data)

    all_samples_triples_label = []
    all_samples_triples_pred = []
    pred_data_i = 0
    for data_i in range(len(label_data)):  # 遍历标签

        # -------------------- process label
        triples_label_with_pos = label_data[data_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                pass
                # triple_info['subj'] = subj_str.split()[-1].lower()   # 实体取最后一个单词
                # # triple_info['subj'] = subj_str,   # 实体取最后一个单词
                # # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                # triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                # triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[data_i]['triple_list_pred'].copy()
        ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str[0]
            rela_str = triple_str[1]
            obj_str = triple_str[2]

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                pass
                # triple_info['subj'] = subj_str.split()[-1].lower()   # 实体取最后一个单词
                # # triple_info['subj'] = subj_str,
                # # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                # # triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                # triple_info['rela'] = rela_str
                # triple_info['obj'] = obj_str.split()[-1].lower()
                # # triple_info['obj'] = obj_str,
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)

        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f"triples_pred{data_i} = {triples_pred}")

        # triples_label_str = [tuple(triple[0]) for triple in triples_label]
        # triples_pred_str = [(triple['subject'], triple['predicate'], triple['object']) for triple in triples_pred_gather]
        # triples_pred_str = list(set(triples_pred_str))
        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    # input format:
    # print(all_samples_triples_pred[10])
    # print(all_samples_triples_label[10])
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple


def evaluate_1_checkpoint__for_SPN(predict_file, label_file, tokenizer):

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(f"len(predict_data['gold']) = {len(predict_data['gold'])}")
    # print(f"len(predict_data['pred']) = {len(predict_data['pred'])}")
    # print(f"len(label_data) = {len(label_data)}")
    # assert len(predict_data['gold']) == len(label_data) and len(predict_data['gold']) == len(predict_data['pred'])
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    for data_i in range(len(label_data)):  # 遍历标签

        # # -------------------- label
        # text_id = label_data[data_i]['id']
        # text = label_data[data_i]['text']
        # triples_label_with_pos = label_data[data_i]['relation_list_original'].copy()
        # ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        # # 处理标签实体
        # triples_label = []
        # for triple_str_pos in triples_label_with_pos:
        #     subj_str_origin = triple_str_pos[0][0]
        #     rela_str = triple_str_pos[0][1]
        #     obj_str_origin = triple_str_pos[0][2]
        #     # print(tokenizer_global.encode(subj_str_origin))
        #     # print(xxxxx)
        #     subj_str = tokenizer.decode(tokenizer.encode(subj_str_origin, add_special_tokens=False))
        #     subj_str = process_after_BertTokenizer_decode(subj_str)
        #     obj_str = tokenizer.decode(tokenizer.encode(obj_str_origin, add_special_tokens=False))
        #     obj_str = process_after_BertTokenizer_decode(obj_str)
        #     triples_label.append((subj_str, rela_str, obj_str))
        triples_label_with_pos = label_data[data_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {}
            triple_info['subj'] = subj_str
            triple_info['rela'] = rela_str
            triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[data_i]['triple_pred_list'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str["subject"]
            rela_str = triple_str["predicate"]
            obj_str = triple_str["object"]

            triple_info = {}
            triple_info['subj'] = subj_str
            triple_info['rela'] = rela_str
            triple_info['obj'] = obj_str
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
        # triples_pred_0 = predict_data['pred'][str(data_i)].copy()
        # tokens_id = tokenizer.encode(text, add_special_tokens=True)
        # triples_pred = []
        # for triple_str_0 in triples_pred_0:
        #     """ example of triple_str_0
        #     [
        #         3, 0.999967098236084,
        #         1, 6, 0.9999997615814209, 1.0,
        #         24, 29, 0.9999912977218628, 0.9999896287918091
        #     ]
        #     """
        #     subj_tokens_id = tokens_id[triple_str_0[2]:triple_str_0[3] + 1]
        #     obj_tokens_id = tokens_id[triple_str_0[6]:triple_str_0[7] + 1]
        #     subj_str = tokenizer.decode(subj_tokens_id)
        #     subj_str = process_after_BertTokenizer_decode(subj_str)
        #     obj_str = tokenizer.decode(obj_tokens_id)
        #     obj_str = process_after_BertTokenizer_decode(obj_str)
        #     rela_str = relation_list[triple_str_0[0]]
        #     triple_str = (subj_str, rela_str, obj_str)
        #     if triple_str not in triples_pred:
        #         triples_pred.append(triple_str)
        #
        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f" triples_pred{data_i} = {triples_pred}")
        #     print("")

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple


def evaluate_1_checkpoint__for_TPLinker(predict_file, label_file, tokenizer):
    def span_have_overlap(span1, span2):
        # 判断是否有重叠
        # 情况1: span1的结束值大于span2的起始值，且span1的起始值小于span2的结束值
        # 情况2: span2的结束值大于span1的起始值，且span2的起始值小于span1的结束值
        # 两种情况之一为真，则存在重叠

        # 提取跨度的起始和结束值
        x1, y1 = span1
        x2, y2 = span2
        return (x1 < y2 and x2 < y1) or (x2 < y1 and x1 < y2)

    MAX_TOKEN_SEQ_LEN = 200
    TOKEN_SLIDING_LEN = 50
    tplinker_dataset_old_version = False  # 数据集格式版本为旧版

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """ format in tplinker output
            {
                "text": "系统消息4包含LAI、RACH控制参数信息。",
                "relation_list_pred": [
                    {
                        "subject": "ServingGW",
                        "object": "SAE-GW",
                        "subj_char_span": [
                            0,
                            9
                        ],
                        "obj_char_span": [
                            18,
                            24
                        ],
                        "predicate": "别名"
                    },
                    {...}, ...
                ]
            },
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    print(len(predict_data), len(label_data))

    # 由于tplinker中有split机制，过长的句子会被拆分为好几条，在此进行合并处理
    for sample_i in range(len(predict_data) - 1, -1, -1):
        if predict_data[sample_i]['tok_offset'] > 0:
            assert predict_data[sample_i]['id'] == predict_data[sample_i - 1]['id'], \
                f"\n{predict_data[sample_i - 1]['id']}\n{predict_data[sample_i]['id']}"

            # # idea1 全部合并到前一个，是否重复交给后续程序判断
            # predict_data[sample_i - 1]['relation_list_pred'] += predict_data[sample_i]['relation_list_pred'].copy()
            # del predict_data[sample_i]

            # idea2 丢弃 offset!=0 的样本（与idea1差不多，为了与其他模型近似，选择该策略）
            del predict_data[sample_i]

            # # idea3 offset!=0 的样本，只取新增范围内的三元组
            # new_sliding_span = (MAX_TOKEN_SEQ_LEN - TOKEN_SLIDING_LEN, 10000)  # 三元组获取范围
            # for triple_pred in predict_data[sample_i]['relation_list_pred']:
            #     if (span_have_overlap(new_sliding_span, triple_pred['subj_tok_span']) or
            #             span_have_overlap(new_sliding_span, triple_pred['obj_tok_span'])):
            #         triple_pred['subj_char_span'][0] += predict_data[sample_i]['char_offset']
            #         triple_pred['subj_char_span'][1] += predict_data[sample_i]['char_offset']
            #         triple_pred['obj_char_span'][0] += predict_data[sample_i]['char_offset']
            #         triple_pred['obj_char_span'][1] += predict_data[sample_i]['char_offset']
            #         triple_pred['subj_tok_span'][0] += predict_data[sample_i]['tok_offset']
            #         triple_pred['subj_tok_span'][1] += predict_data[sample_i]['tok_offset']
            #         triple_pred['obj_tok_span'][0] += predict_data[sample_i]['tok_offset']
            #         triple_pred['obj_tok_span'][1] += predict_data[sample_i]['tok_offset']
            #         predict_data[sample_i - 1]['relation_list_pred'].append(triple_pred.copy())
            # del predict_data[sample_i]

    # 由于tplinker中有split机制，过长的句子会被拆分为好几条，在此进行合并处理
    ########################################

    assert len(predict_data) == len(label_data), f"\n{len(predict_data)}\n{len(label_data)}"

    all_samples_triples_label = []
    all_samples_triples_pred = []
    pred_data_i = 0
    for data_i in range(len(label_data)):  # 遍历标签

        # -------------------- process prediction
        triples_pred_in = predict_data[data_i]['relation_list_pred'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str["subject"]
            rela_str = triple_str["predicate"]
            obj_str = triple_str["object"]

            triple_info = {}
            triple_info['subj'] = subj_str
            triple_info['rela'] = rela_str
            triple_info['obj'] = obj_str
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)

        # -------------------- process label
        if tplinker_dataset_old_version:
            triples_label_with_pos = label_data[data_i]['relation_list_original'].copy()
        else:
            triples_label_with_pos = label_data[data_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            if tplinker_dataset_old_version:
                subj_str = triple_str_pos[0][0]
                rela_str = triple_str_pos[0][1]
                obj_str = triple_str_pos[0][2]
                subj_char_span = triple_str_pos[1]
                obj_char_span = triple_str_pos[2]
            else:
                subj_str = triple_str_pos['subject']
                rela_str = triple_str_pos['predicate']
                obj_str = triple_str_pos['object']
                subj_char_span = triple_str_pos['subj_char_span']
                obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {}
            triple_info['subj'] = subj_str
            triple_info['rela'] = rela_str
            triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    # input format:
    # print(all_samples_triples_pred[10])
    # print(all_samples_triples_label[10])
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple


def score_all():
    # # pretrain model
    # config = AutoConfig.from_pretrained(args_from_yaml['model_name_or_path'])
    # print(f"pretain model config = {config}\n")
    #
    # tokenizer = T5Tokenizer.from_pretrained(args_from_yaml['model_name_or_path'], use_fast=False)
    # special_tokens_list = args_from_yaml['special_tokens']
    # if len(special_tokens_list) > 0:
    #     tokenizer.add_tokens(special_tokens_list)

    if args_global.MODEL_NAME == 'ours':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_ourmodel
    elif args_global.MODEL_NAME == 'BiRTE':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_BiRTE
    elif args_global.MODEL_NAME.lower() == 'tplinker':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_TPLinker
    elif args_global.MODEL_NAME == 'SPN':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_SPN

    # tokenizer
    tokenizer = None
    if 1:  # args_global.MODEL_NAME in ['ours', 'BiRTE', 'tplinker', 'SPN']:
        from transformers import BertTokenizer
        # from dataset_loader import ADDITIONAL_SPECIAL_TOKENS
        tokenizer = BertTokenizer.from_pretrained(args_global.PRETRAIN_MODEL_DIR, do_lower_case=True)
        ADDITIONAL_SPECIAL_TOKENS = ['“', '”']
        tokenizer.add_tokens(ADDITIONAL_SPECIAL_TOKENS)

    # get checkpoint dir
    checkpoint_dir_list = []
    folders = list(os.walk(args_global.OUTPUT_DIR))[0][1]
    for folder in folders:
        if args_global.CHECKPOINT_FOLDER_PREFIX in folder:
            checkpoint_dir_list.append(os.path.join(args_global.OUTPUT_DIR, folder))
    checkpoint_dir_list.sort(
        key=lambda x: str_left_add(suffix_find(x, args_global.CHECKPOINT_FOLDER_PREFIX), "0", 9))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(5)

    res_all = []
    for checkpoint_dir in checkpoint_dir_list:
        res = {
            'step': checkpoint_dir,
        }
        print(f"\nstep={checkpoint_dir}")

        predict_file = os.path.join(checkpoint_dir, args_global.PREDICT_FILENAME_dev)
        label_file = os.path.join(args_global.DATASET_DIR, args_global.LABEL_FILENAME_dev)
        if os.path.isfile(predict_file) is True:
            res_triple_dev = evaluate_1_checkpoint(
                predict_file=predict_file,
                label_file=label_file,
                tokenizer=tokenizer
            )
            res['triple_dev'] = res_triple_dev
            print(f"triple dev: {res_triple_dev}")

        predict_file = os.path.join(checkpoint_dir, args_global.PREDICT_FILENAME_test)
        label_file = os.path.join(args_global.DATASET_DIR, args_global.LABEL_FILENAME_test)
        if os.path.isfile(predict_file) is True:
            res_triple_test = evaluate_1_checkpoint(
                predict_file=predict_file,
                label_file=label_file,
                tokenizer=tokenizer,
            )
            res['triple_test'] = res_triple_test
            print(f"triple test: {res_triple_test}")
        # res_triple_test = evaluate_1_checkpoint_240424(os.path.join(output_dir, "dataset_prediction_test.json"), ask_num, tokenizer)
        # res['triple_test'] = res_triple_test
        # print(f"triple test: {res_triple_test}")

        # delete bad model
        model_file = os.path.join(checkpoint_dir, args_global.CHECKPOINT_MODEL_NAME)
        if args_global.DELETE_BAD_MODEL is True and os.path.isfile(model_file) is True \
                and res.get('triple_dev') is not None and res['triple_dev']['f1'] < args_global.BAD_MODEL_THRE:
            os.remove(model_file)
            print(f"模型效果差，已删除参数文件：{model_file}")
            time.sleep(5)
        # else:
        #     print(DELETE_BAD_MODEL)
        #     print(os.path.isfile(model_file))
        #     print(res.get('triple_dev'))
        #     print(os.path.isfile(model_file))
        #     time.sleep(3)

        res_all.append(res)

    res_all.sort(key=lambda x: x['triple_dev']['f1'], reverse=True)
    res_all_TestTop = res_all.copy()
    res_all_TestTop.sort(key=lambda x: x['triple_test']['f1'], reverse=True)  # 即，将dev、test角色颠倒

    # 计算 valid 与 test f1 之间差值的平均值
    res_top10 = res_all[:10].copy() if len(res_all) > 10 else res_all.copy()
    f1_dif = 0
    for res in res_top10:
        f1_dif += res['triple_test']['f1'] - res['triple_dev']['f1']
    f1_dif /= len(res_top10)
    print(f"f1_dif_between_dev_test = {f1_dif}")
    res_all.append({'aver_top10_f1_dif': f1_dif})

    rouge_suffix = "complete"
    if args_global.USE_ROUGE:
        rouge_suffix = f"{args_global.WHICH_ROUGE.replace('-', '')}" \
                       f"({args_global.ROUGE_THRE},{args_global.TOKEN_TYPE})"

    file_name = f"score_triple_{rouge_suffix}.json"
    file_name_TestTop = f"score_triple(TestTop)_{rouge_suffix}.json"

    with open(os.path.join(args_global.OUTPUT_DIR, file_name), "w", encoding="utf-8") as fp:
        json.dump(res_all, fp, ensure_ascii=False, indent=4)
    # with open(os.path.join(args_global.OUTPUT_DIR, file_name_TestTop), "w", encoding="utf-8") as fp:
    #     json.dump(res_all_TestTop, fp, ensure_ascii=False, indent=4)
    print(f"score_triple.json saved")
    return os.path.join(args_global.OUTPUT_DIR, file_name)


def paint_f1_curve(file_path_name):
    print("paint curve")

    file_path = os.path.dirname(file_path_name)
    file_name = os.path.basename(file_path_name)
    print("file_path: ", file_path)
    print("file_name: ", file_name)
    time.sleep(5)

    with open(file_path_name, "r", encoding="utf-8") as fp:
        res_all = json.load(fp)
    for i in range(len(res_all) - 1, -1, -1):
        if res_all[i].get('step') is None:
            del res_all[i]  # {'aver_top10_f1_dif': -0.0204}
    res_all.sort(key=lambda x: str_left_add(suffix_find(x['step'], args_global.CHECKPOINT_FOLDER_PREFIX), "0", 9))

    for res in res_all:
        print("  ", res['step'])
    print("check the order")
    time.sleep(5)

    step_number_list = []
    dev_f1_list = []
    test_f1_list = []
    for res in res_all:
        step_number_list.append(int(suffix_find(res['step'], args_global.CHECKPOINT_FOLDER_PREFIX)))
        dev_f1_list.append(res['triple_dev']['f1'])
        test_f1_list.append(res['triple_test']['f1'])

    # paint
    plt.plot(step_number_list, dev_f1_list, label='dev_f1', color='blue')
    plt.plot(step_number_list, test_f1_list, label='test_f1', color='red')
    plt.legend()  # 添加图例
    plt.title(f"{file_name} f1 change")
    plt.xlabel('step')
    plt.ylabel('f1')
    plt.grid(True)  # 网格线
    pic_name = "curve(f1)_" + file_name[:-5] + ".png"  # json -> png
    plt.savefig(os.path.join(file_path, pic_name))
    print("curve saved")


if __name__ == "__main__":
    print(args_global)
    time.sleep(5)
    timepoint_start = time.time()
    res_path_name = score_all()
    print(f"get score END. take time: {(time.time() - timepoint_start) / 60} mins")

    # res_path_name = "outputs/LSTM compare in LossCRF/240725_LSTM-BiL2H576_LossCRF/score_triple_complete.json"
    paint_f1_curve(res_path_name)

    # # test. 获取rouge分数
    # get_rouge_test(
    #     pred_txt="支持GPRS的CS3和CS4编码，实现信道编码功能",
    #     label_txt="支持GPRS的CS3和CS4编码",)
