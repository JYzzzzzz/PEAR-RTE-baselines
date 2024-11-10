import os
import time
import json
import yaml

import jieba
import rouge_chinese


OUTPUT_DIR = "output/240613"
DATASET_DIR = "data4bert/CMIM2023_KG_task1_re/240607_seed0"

# tokenizer
from transformers import BertTokenizerFast
from dataprocess.data_processor import token_decode
MODEL_DIR = "model/chinese-bert-wwm-ext"
tokenizer_global = BertTokenizerFast.from_pretrained(
        MODEL_DIR, do_basic_tokenize=False,
        add_special_tokens=True, do_lower_case=True
    )

# rouge
USE_ROUGE = True
WHICH_ROUGE = "rouge-1"
ROUGE_THRE = 0.6


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
        sub_str = sub_str[symbol_l_pos2+len(symbol_l_str):]
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


def get_rouge(pred_txt: str, label_txt: str):  # 获取rouge分数
    pred_tokens = list(jieba.cut(pred_txt))
    label_tokens = list(jieba.cut(label_txt))
    rouge = rouge_chinese.Rouge()
    scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
    """
    [{
        'rouge-1': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}, 
        'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
        'rouge-l': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}
    }]
    """
    return scores


def rouge_triple_in_labels(triple_pred, triples_label_remain: list):
    """
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
        if triple_pred[1] != triple_label[1]:
            continue
        # 获取 pred、label 主体间的 rouge
        subj_rouge_score = get_rouge(triple_pred[0], triple_label[0])[0]
        # 客体间的 rouge
        obj_rouge_score = get_rouge(triple_pred[2], triple_label[2])[0]
        # 融合rouge分数的策略1: 主客体 WHICH_ROUGE 的较小值
        triple_score = min(subj_rouge_score[WHICH_ROUGE]['f'], obj_rouge_score[WHICH_ROUGE]['f'])
        # 更新best
        if triple_score > best_label['rouge_score']:
            best_label['rouge_score'] = triple_score
            best_label['triple_i'] = triple_i
            if best_label['rouge_score'] > 0.99:   # == 1.0
                break

    if best_label['rouge_score'] > ROUGE_THRE:   # 阈值
        del triples_label_remain[best_label['triple_i']]
        return True
    else:
        return False


def f1_score_triple(preds: list, labels: list):
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

        if USE_ROUGE is False:
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
                if rouge_triple_in_labels(triple_pred, triples_label_remain):
                    assert len(triples_label_remain) == len_triples_label_remain - 1
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


def evaluate_1_checkpoint(predict_file, label_file,):

    def span_have_overlap(span1, span2):
        # 判断是否有重叠
        # 情况1: span1的结束值大于span2的起始值，且span1的起始值小于span2的结束值
        # 情况2: span2的结束值大于span1的起始值，且span2的起始值小于span1的结束值
        # 两种情况之一为真，则存在重叠

        # 提取跨度的起始和结束值
        x1, y1 = span1
        x2, y2 = span2
        return (x1 < y2 and x2 < y1) or (x2 < y1 and x1 < y2)

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    print(len(predict_data), len(label_data))
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    pred_data_i = 0
    for data_i in range(len(label_data)):   # 遍历标签
        text_id = label_data[data_i]['id']
        text = label_data[data_i]['text']
        triples_label_with_pos = label_data[data_i]['relation_list_original'].copy()
        ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        # 处理标签实体
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str_origin = triple_str_pos[0][0]
            rela_str = triple_str_pos[0][1]
            obj_str_origin = triple_str_pos[0][2]
            # print(tokenizer_global.encode(subj_str_origin))
            # print(xxxxx)
            subj_str = token_decode(tokenizer_global, tokenizer_global.encode(subj_str_origin))
            obj_str = token_decode(tokenizer_global, tokenizer_global.encode(obj_str_origin))
            triples_label.append([subj_str, rela_str, obj_str])

        # 同步收集 预测
        triples_pred = predict_data[data_i]['pred_spo_list'].copy()
        ##### [[subj,rela,obj], [...], ...]

        if data_i < 3:
            print(f"triples_label{data_i} = {triples_label}")
            print(f"triples_pred{data_i} = {triples_pred}")

        # triples_label_str = [tuple(triple[0]) for triple in triples_label]
        # triples_pred_str = [(triple['subject'], triple['predicate'], triple['object']) for triple in triples_pred_gather]
        # triples_pred_str = list(set(triples_pred_str))
        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label)
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

    # get checkpoint dir
    checkpoint_dir_list = []
    folders = list(os.walk(OUTPUT_DIR))[0][1]
    for folder in folders:
        if 'checkpoint' in folder:
            checkpoint_dir_list.append(os.path.join(OUTPUT_DIR, folder))
    checkpoint_dir_list.sort(key=lambda x: str_left_add(suffix_find(x, 'checkpoint-'), "0", 9))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(5)

    res_all = []
    for checkpoint_dir in checkpoint_dir_list:
        res = {
            'step': checkpoint_dir,
        }
        print(f"\nstep={checkpoint_dir}")

        predict_file = os.path.join(checkpoint_dir, "dev_predict_sard.json")
        label_file = os.path.join(DATASET_DIR, "valid_data.json")
        if os.path.isfile(predict_file) is True:
            res_triple_dev = evaluate_1_checkpoint(
                predict_file=predict_file,
                label_file=label_file,)
            res['triple_dev'] = res_triple_dev
            print(f"triple dev: {res_triple_dev}")

        predict_file = os.path.join(checkpoint_dir, "test_predict_sard.json")
        label_file = os.path.join(DATASET_DIR, "test_data.json")
        if os.path.isfile(predict_file) is True:
            res_triple_test = evaluate_1_checkpoint(
                predict_file=predict_file,
                label_file=label_file,)
            res['triple_test'] = res_triple_test
            print(f"triple test: {res_triple_test}")
        # res_triple_test = evaluate_1_checkpoint_240424(os.path.join(output_dir, "dataset_prediction_test.json"), ask_num, tokenizer)
        # res['triple_test'] = res_triple_test
        # print(f"triple test: {res_triple_test}")

        res_all.append(res)

    res_all.sort(key=lambda x: x['triple_dev']['f1'], reverse=True)

    # 计算 valid 与 test f1 之间差值的平均值
    res_top10 = res_all[:10].copy() if len(res_all) > 10 else res_all.copy()
    f1_dif = 0
    for res in res_top10:
        f1_dif += res['triple_test']['f1'] - res['triple_dev']['f1']
    f1_dif /= len(res_top10)
    print(f"f1_dif_between_dev_test = {f1_dif}")
    res_all.append({'aver_top10_f1_dif': f1_dif})

    if USE_ROUGE:
        file_name = f"score_triple_{WHICH_ROUGE.replace('-', '')}_{ROUGE_THRE}.json"
    else:
        file_name = f"score_triple_complete.json"
    with open(os.path.join(OUTPUT_DIR, file_name), "w", encoding="utf-8") as fp:
        json.dump(res_all, fp, ensure_ascii=False, indent=4)
        print(f"score_triple.json saved")
    

if __name__ == "__main__":
    score_all()
