import os
import re
import unicodedata
import math
import json
import random
import copy
import numpy as np
from tqdm import tqdm
from utils import load_json, load_dict, write_dict, str_q2b
import dataprocess.rel2text

from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def save_dict(dict, name):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(f'{name}.txt', 'w', encoding='utf-8') as f:
        f.write(str(dict))  # dict to str

def remove_stress_mark(text):
    text = "".join([c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"])
    return text
 
def change_case(str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return re.sub(r'[^\w\s]','',s2)


def token_decode(tokenizer, token_ids):    # jyz chg 2024-06
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        run_args.model_dir, additional_special_tokens=added_token, do_basic_tokenize=False,
        add_special_tokens=True, do_lower_case=True)
    以上 tokenizer decode 时，汉字之间都会有空格，开头的##不会消除，因此进行手动处理
    """
    a2z = "abcdefghijklmnopqrstuvwxyz"
    text_decode = tokenizer.decode(token_ids)
    text = ""
    # 手动去除token间的空格，但保留英文单词间的
    for i in range(len(text_decode)):
        if text_decode[i] == " ":
            if text_decode[i-1] in a2z and text_decode[i+1] in a2z:
                text += text_decode[i]
        else:
            text += text_decode[i]
    # 去除首尾的特殊字符
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.strip("#")
    return text


# Driver code
class UniRelDataProcessor(object):
    def __init__(self,
                 root,
                 tokenizer,
                 is_lower=False,
                 dataset_name='nyt',
                 ):
        self.task_data_dir = os.path.join(root, dataset_name)
        self.train_path = os.path.join(self.task_data_dir, 'train_split.json')  # 数据集路径
        self.dev_path = os.path.join(self.task_data_dir, 'valid_data.json')
        self.test_path = os.path.join(self.task_data_dir, 'test_data.json')

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        self.label_map_cache_path = os.path.join(self.task_data_dir,
                                                 dataset_name + '.dict')

        self.label2id = None
        self.id2label = None
        self.max_label_len = 0

        self._get_labels()

        if dataset_name == "nyt":   # run_nyt.sh into this
            self.pred2text=dataprocess.rel2text.nyt_rel2text
            ##### “数据集原关系文字描述” 到 “模型使用的关系文字描述” 的映射
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}

        elif "CMIM" in dataset_name:  # jyz chg. 2024-06
            self.pred2text = dataprocess.rel2text.CMIM2023_KG_task1_re__rel2text

        elif dataset_name == "nyt_star":
            self.pred2text=dataprocess.rel2text.nyt_rel2text
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        elif dataset_name == "webnlg":
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
            self.pred2text=dataprocess.rel2text.webnlg_rel2text
            cnt = 1
            exist_value=[]
            # Some hard to convert relation directly use [unused]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
        elif dataset_name == "webnlg_star":
            self.pred2text={}
            for pred in self.label2id.keys():
                try:
                    self.pred2text[pred] = dataprocess.rel2text.webnlg_rel2text[pred]
                except KeyError:
                    print(pred)
            cnt = 1
            exist_value=[]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}

        self.num_rels = len(self.pred2text.keys())
        self.max_label_len = 1
        self.pred2idx = {}     # 存放关系（谓语）到id的映射
        idx = 0
        self.pred_str = ""
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1  # idx从0累加
        self.pred_str = self.pred_str[:-1]
        print(self.pred_str)
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        """
            self.pred2text = {'/people/person/children': 'children', '/people/person/place_lived': 'lived', ...}
            self.pred2idx = {'/people/person/children': 0, '/people/person/place_lived': 1, ...}
            self.pred_str = "children lived ..."
            self.idx2pred = {0:'/people/person/children', 1:'/people/person/place_lived', ...}
        """

        self.num_labels = self.num_rels

        # jyz chg 2024-06. 用于中文数据集生成准确的token_span，不考虑起始特殊符号[CLS]等
        self.data_span_converter = Char_Token_SpanConverter(tokenizer=tokenizer)

    def get_train_sample(self, token_len=100, data_nums=-1):  # used in run.py
        return self._pre_process(self.train_path,
                                 token_len=token_len,
                                 is_predict=False,
                                 data_nums=data_nums)

    def get_dev_sample(self, token_len=150, data_nums=-1):
        return self._pre_process(self.dev_path,
                                 token_len=token_len,
                                 is_predict=True,
                                 data_nums=data_nums)

    def get_test_sample(self, token_len=150, data_nums=-1):
        samples = self._pre_process(self.test_path,
                                    token_len=token_len,
                                    is_predict=True,
                                    data_nums=data_nums)
        # json.dump(self.complex_data, self.wp, ensure_ascii=False)
        return samples

    def get_specific_test_sample(self, data_path, token_len=150, data_nums=-1):
        return self._pre_process(data_path,
                                 token_len=token_len,
                                 is_predict=True,
                                 data_nums=data_nums)

    def _get_labels(self):
        label_num_dict = {}
        # if os.path.exists(self.label_map_cache_path):
        #     label_map = load_dict(self.label_map_cache_path)
        # else:
        label_set = set()
        for path in [self.train_path, self.dev_path, self.test_path]:
            fp = open(path)
            samples = json.load(fp)
            for data in samples:
                sample = data
                for spo in sample["relation_list"]:
                    label_set.add(spo["predicate"])
                    if spo["predicate"] not in label_num_dict:
                        label_num_dict[spo["predicate"]] = 0
                    label_num_dict[spo["predicate"]] += 1
        label_set = sorted(label_set)
        labels = list(label_set)
        label_map = {idx: label for idx, label in enumerate(labels)}
        # write_dict(self.label_map_cache_path, label_map)
        # fp.close()
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, token_len, is_predict, data_nums):
        outputs = {
            'text': [],
            "spo_list": [],
            "spo_span_list": [],
            "head_label": [],
            "tail_label": [],
            "span_label": []
        }
        token_len_big_than_100 = 0
        token_len_big_than_150 = 0
        max_token_len = 0
        max_data_nums = math.inf if data_nums == -1 else data_nums
        data_count = 0
        datas = json.load(open(path))
        label_dict = {}

        for sample in datas:    # 遍历样本

            text = sample['text']
            triples = sample['relation_list']
            entities = []

            # if len(triples) == 0:  # 不存在关系三元组，跳过不作为样本呢
            #     continue
            ##### jyz chg 2024-06. 因为我的中文数据集中有些样本确实没有任何关系

            # 统计句子tokens长度
            input_ids = self.tokenizer.encode(sample["text"])
            token_encode_len = len(input_ids)
            if token_encode_len > 100+2:
                token_len_big_than_100 += 1
            if token_encode_len > 150+2:
                token_len_big_than_150 += 1
            max_token_len = max(max_token_len, token_encode_len)
            # if token_encode_len > token_len + 2:   # 如果token长度大于阈值，跳过？？？？？？
            #     continue
            # ^^^ jyz chg 2024-06

            # jyz chg 2024-09. 调整带分段实体的三元组 - 删除
            for i in range(len(triples)-1, -1, -1):
                if len(triples[i]['subj_char_span']) > 1 or len(triples[i]['obj_char_span']) > 1:
                    del triples[i]
                else:
                    triples[i]['subj_char_span'] = tuple(triples[i]['subj_char_span'][0].copy())
                    triples[i]['obj_char_span'] = tuple(triples[i]['obj_char_span'][0].copy())
                    assert len(triples[i]['subj_char_span']) == 2, f"\n{triples}"

            # jyz chg 2024-06. 调整过长的句子
            if token_encode_len > token_len + 2:
                while token_encode_len > token_len + 2:
                    sample["text"] = sample["text"][:-10]
                    input_ids = self.tokenizer.encode(sample["text"])
                    token_encode_len = len(input_ids)
                # 删除在句子长度之外的三元组
                list_temp = []
                for triple in triples:
                    if max(triple['subj_char_span'][1],
                           triple['obj_char_span'][1]) <= len(sample["text"]):
                        list_temp.append(triple.copy())
                triples = list_temp.copy()

            # jyz chg 2024-06. 重新添加 tok_span，
            for i in range(len(triples)):
                triples[i]['subj_tok_span'] = self.data_span_converter.get_tok_span(
                    text, triples[i]['subj_char_span'])
                triples[i]['obj_tok_span'] = self.data_span_converter.get_tok_span(
                    text, triples[i]['obj_char_span'])
            # for i in range(len(sample['entity_list'])):
            #     sample['entity_list'][i]['tok_span'] = self.data_span_converter.get_tok_span(
            #         text, sample['entity_list'][i]['char_span'])

            # jyz chg 2024-09. 添加 entity_list
            """ format
            {
                "text": "SAE-GW",
                "type": "Default",
                "char_span": [ 18, 24 ],
                "tok_span": [ 6, 9 ]
            },
            """
            for triple in triples:
                entity = {
                    "text": triple['subject'],
                    "type": "Default",
                    "char_span": triple['subj_char_span'],
                    "tok_span": triple['subj_tok_span']
                }
                if entity not in entities:
                    entities.append(entity.copy())
                entity = {
                    "text": triple['object'],
                    "type": "Default",
                    "char_span": triple['obj_char_span'],
                    "tok_span": triple['obj_tok_span']
                }
                if entity not in entities:
                    entities.append(entity.copy())
            # for entity_info in sample['entity_list']:
            #     if entity_info['char_span'][1] <= len(sample["text"]):
            #         list_temp.append(entity_info)
            # sample['entity_list'] = list_temp.copy()

            # init for outputs. 存放每个样本各自的信息
            spo_list = set()
            spo_span_list = set()
            # 矩阵尺寸构成：[CLS] texts [SEP] rels
            head_matrix = np.zeros([token_len + 2 + self.num_rels,
                                    token_len + 2 + self.num_rels])
            tail_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels])
            span_matrix = np.zeros(
                [token_len + 2 + self.num_rels, token_len + 2 + self.num_rels])

            e2e_set = set()
            h2r_dict = dict()
            t2r_dict = dict()
            spo_tail_set = set()
            spo_tail_text_set = set()
            spo_text_set = set()
            for spo in triples:
                # spo means triple(subject, predicate, object)

                predicate = spo["predicate"]
                if predicate not in label_dict:  # 如果该关系还未在统计中出现，则设置数量为0
                    label_dict[predicate] = 0
                label_dict[predicate] += 1
                sub = spo["subject"]
                obj = spo["object"]
                spo_list.add((sub, predicate, obj))   # triple text set

                sub_span = spo["subj_tok_span"]
                obj_span = spo["obj_tok_span"]
                pred_idx = self.pred2idx[predicate]
                plus_token_pred_idx = pred_idx + token_len + 2  # predicate 在矩阵中的 idx
                spo_span_list.add((tuple(sub_span), pred_idx, tuple(obj_span)))
                ##### set of (subj_tok_span, predicate_idx, obj_tok_span)

                sub_s, sub_e = sub_span
                obj_s, obj_e = obj_span
                # Entity-Entity Interaction
                head_matrix[sub_s+1][obj_s+1] = 1     # +1 because of [CLS]
                head_matrix[obj_s+1][sub_s+1] = 1
                tail_matrix[sub_e][obj_e] = 1
                tail_matrix[obj_e][sub_e] = 1
                span_matrix[sub_s+1][sub_e] = 1
                span_matrix[sub_e][sub_s+1] = 1
                span_matrix[obj_s+1][obj_e] = 1
                span_matrix[obj_e][obj_s+1] = 1
                # Subject-Relation Interaction
                head_matrix[sub_s+1][plus_token_pred_idx] = 1
                tail_matrix[sub_e][plus_token_pred_idx] = 1
                span_matrix[sub_s+1][plus_token_pred_idx] = 1
                span_matrix[sub_e][plus_token_pred_idx] = 1
                span_matrix[obj_s+1][plus_token_pred_idx] = 1
                span_matrix[obj_e][plus_token_pred_idx] = 1
                # Relation-Object Interaction
                head_matrix[plus_token_pred_idx][obj_s+1] = 1
                tail_matrix[plus_token_pred_idx][obj_e] = 1
                span_matrix[plus_token_pred_idx][obj_s+1] = 1
                span_matrix[plus_token_pred_idx][obj_e] = 1
                span_matrix[plus_token_pred_idx][sub_s+1] = 1
                span_matrix[plus_token_pred_idx][sub_e] = 1

                # useless, maybe for check
                spo_tail_set.add((sub_e, plus_token_pred_idx, obj_e))
                spo_tail_text_set.add((
                    self.tokenizer.decode(input_ids[sub_e]),
                    predicate,
                    self.tokenizer.decode(input_ids[obj_e])
                ))
                spo_text_set.add((
                    self.tokenizer.decode(input_ids[sub_s+1:sub_e+1]),
                    predicate,
                    self.tokenizer.decode(input_ids[obj_s+1:obj_e+1])
                ))
                e2e_set.add((sub_e, obj_e))
                e2e_set.add((obj_e, sub_e))
                # loop `for spo in triples:` end

            outputs["text"].append(sample["text"])
            outputs["spo_list"].append(list(spo_list))
            outputs["spo_span_list"].append(list(spo_span_list))
            outputs["head_label"].append(head_matrix)
            outputs["tail_label"].append(tail_matrix)
            outputs["span_label"].append(span_matrix)

            data_count += 1
            if data_count >= max_data_nums:   # 大于设定的最大样本数，停止，退出
                break
            # loop `for sample in tqdm(data):` end

        print(max_token_len)
        print(f"more than 100: {token_len_big_than_100}")
        print(f"more than 150: {token_len_big_than_150}")
        return outputs


class Char_Token_SpanConverter(object):
    """
    用于数据集生成准确的 token_char_mapping, 并互化
    version 240725 : 考虑了span互化时，输入span为(x,x)的异常情况，print了一些提示信息。
    version 240825: 添加 返回mapping的函数
    """

    def __init__(self, tokenizer, add_special_tokens=False, has_return_offsets_mapping=True):
        """
        add_special_tokens: 如果 add_special_tokens=True，会将 [CLS] 考虑在内，token_span 数值整体+1
        has_return_offsets_mapping: bool. tokenizer自身是否包含return_offsets_mapping功能，若不包含，由spanconverter生成。
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


# class Char_Token_SpanConverter(object):   # jyz chg 2024-06. 用于中文数据集生成准确的token_span
#     def __init__(self, tokenizer, add_special_tokens=False, has_return_offsets_mapping=True):
#         """
#         add_special_tokens: 如果 add_special_tokens=True，会将 [CLS] 考虑在内，token_span 数值整体+1
#         has_return_offsets_mapping: bool. tokenizer是否包含return_offsets_mapping功能，若不包含，手动生成。
#         """
#         self.tokenizer = tokenizer
#         self.token_info = None
#         self.error_tok_spans = []  # {text, char_span, char_span_str, tok_span_str}
#         self.token_decoder = token_decode   # 一般为 tokenizer.decode，但有些可能会有特殊操作，因此设置一个函数指针
#         self.add_special_tokens = add_special_tokens  # 不影响 tokenizer 初始化时设置的 add_special_tokens
#         self.has_return_offsets_mapping = has_return_offsets_mapping
#
#     def get_tok_span(self, text: str, char_span):
#         # get mapping
#         if self.token_info is not None and self.token_info["text"] == text:
#             pass
#         else:
#             self._get_mapping(text)
#         # get token span
#         token_span = self._get_tok_span(char_span)
#         # check
#         self._char_tok_span_check(char_span, token_span)
#         return token_span
#
#     def get_char_span(self, text: str, token_span):
#         # get mapping
#         if self.token_info is not None and self.token_info["text"] == text:
#             pass
#         else:
#             self._get_mapping(text)
#         # get char span
#         char_span_list = self.token_info["tok2char_mapping"][token_span[0]:token_span[1]]
#         char_span = [char_span_list[0][0], char_span_list[-1][1]]
#         return tuple(char_span)
#
#     def _get_mapping(self, text):
#         """
#         实际返回 encode_plus 生成的 token相关信息，其中添加了一些key，主要包括 char2tok_mapping
#         """
#         if self.has_return_offsets_mapping is True:
#             # tok2char_span = _get_tok2char_span_map(text)
#             token_info = self.tokenizer.encode_plus(text,
#                                                return_offsets_mapping=True,
#                                                add_special_tokens=self.add_special_tokens)
#             token_info["text"] = text  # 添加原文
#             token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])
#
#             tok2char_span = token_info["offset_mapping"]
#             token_info["tok2char_mapping"] = tok2char_span.copy()
#             del token_info["offset_mapping"]
#
#             char_num = None
#             for tok_ind in range(len(tok2char_span) - 1, -1, -1):
#                 if tok2char_span[tok_ind][1] != 0:
#                     char_num = tok2char_span[tok_ind][1]
#                     break
#             char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
#             for tok_ind, char_sp in enumerate(tok2char_span):
#                 for char_ind in range(char_sp[0], char_sp[1]):
#                     tok_sp = char2tok_span[char_ind]
#                     # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
#                     if tok_sp[0] == -1:
#                         tok_sp[0] = tok_ind
#                     tok_sp[1] = tok_ind + 1
#             token_info["char2tok_mapping"] = char2tok_span.copy()
#
#         else:  # self.has_return_offsets_mapping is False
#             token_info = self.tokenizer.encode_plus(text,
#                                                add_special_tokens=self.add_special_tokens)
#             token_info["text"] = text  # 添加原文
#             token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])
#
#             # ---------------------------------------- get char2tok_mapping
#             tokens = token_info["tokens"].copy()
#             char2tok_mapping = [(-1, -1)] * len(text)
#             tokens_i = [0, 0]  # 起始：下标为0的token的下标为0的字符
#             if tokens[0] == self.tokenizer.cls_token:
#                 tokens_i = [1, 0]  # 起始：下标为1的token的下标为0的字符
#             # 遍历字符
#             for c_i, c in enumerate(text):
#                 c_tokens = self.tokenizer.tokenize(c)
#                 if len(c_tokens) == 0:  # c 是一个空白字符
#                     pass
#                 else:
#                     ct = c_tokens[0]
#                     # 查找字符在哪个token中
#                     while ct not in tokens[tokens_i[0]]:
#                         tokens_i[0] += 1
#                         tokens_i[1] = 0
#                         assert tokens_i[0] < len(tokens), f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
#                     # 查找字符在token中哪个位置
#                     ct_pos = tokens[tokens_i[0]].find(ct, tokens_i[1])
#                     assert ct_pos >= tokens_i[1], f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
#                     # 添加到char2tok_mapping
#                     char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
#                     # 更新tokens_i
#                     tokens_i[1] = ct_pos + len(ct)
#                     if tokens_i[1] >= len(tokens[tokens_i[0]]):
#                         tokens_i[0] += 1
#                         tokens_i[1] = 0
#             token_info["char2tok_mapping"] = char2tok_mapping.copy()
#
#             # ---------------------------------------- get tok2char_mapping
#             tok2char_mapping = [(-1, -1)] * len(tokens)
#             for c_i in range(len(text)):
#                 if char2tok_mapping[c_i][0] == -1 or char2tok_mapping[c_i][0] == char2tok_mapping[c_i][1]:
#                     continue
#                 token_i = char2tok_mapping[c_i][0]
#                 if tok2char_mapping[token_i] == (-1, -1):
#                     tok2char_mapping[token_i] = (c_i, c_i + 1)
#                 else:
#                     assert c_i + 1 > tok2char_mapping[token_i][1]
#                     tok2char_mapping[token_i] = (tok2char_mapping[token_i][0], c_i + 1)
#             token_info["tok2char_mapping"] = tok2char_mapping.copy()
#
#         self.token_info = token_info
#         # return token_info
#
#     def _get_tok_span(self, char_span):
#         """
#         得到 tok_span
#         """
#         # char2tok_span: 列表，每个元素表示每个句中字符对应的token下标。
#         #   每个元素一般取值为[a,a+1]，
#         #   如果连续多个元素位于一个token中，则会出现`[a,a+1],[a,a+1],...`，
#         #   如果是例如空格等字符，不会出现在token中，则取值[-1,-1]
#
#         tok_span_list = self.token_info["char2tok_mapping"][char_span[0]:char_span[1]]
#         tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
#         return tok_span
#
#     def _char_tok_span_check(self, char_span, tok_span):
#         """
#         校验 tok_span 是否能抽取出与 char_span 一样的文本
#         token_info: 必须包含 text, input_ids
#         tokenizer: 必须是生成 token_info 的 tokenizer
#         char_span: 长度为2的列表或元组，暂时不考虑分段情况
#         tok_span: 长度为2的列表或元组，暂时不考虑分段情况
#         """
#         sub_text_from_char0 = self.token_info['text'][char_span[0]:char_span[1]]
#         sub_text_from_char = self.token_decoder(self.tokenizer, self.tokenizer.encode(sub_text_from_char0, add_special_tokens=False))
#         sub_text_from_token = self.token_decoder(self.tokenizer, self.token_info['input_ids'][tok_span[0]:tok_span[1]])
#         if sub_text_from_char == sub_text_from_token:
#             return True
#         else:
#             error_tok_span = {
#                 'text': self.token_info['text'],
#                 'char_span': char_span,
#                 'char_span_str': sub_text_from_char,
#                 'tok_span_str': sub_text_from_token
#             }
#             if error_tok_span not in self.error_tok_spans:
#                 self.error_tok_spans.append(error_tok_span)
#                 print(f"char_span string: [{sub_text_from_char0}][{sub_text_from_char}], but tok_span string: [{sub_text_from_token}]")
#             return False

