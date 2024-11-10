# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain

from utils import Label2IdxSub, Label2IdxObj


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
            # print("")
            # print(self.token_info['tokens'])
            # print(token_span[0], token_span[1], char_span_list)
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


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens    # {rela_id1: [(subj, obj), (subj, obj), ...], ...}


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []

    # read src data
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['triple_list']:
                # example of triple: [ "Henry B. Gonzalez", "/people/person/place_of_birth", "San Antonio" ]
                en_pair_list.append([triple[0], triple[-1]])   # 实体对列表
                re_list.append(rel2idx[triple[1]])      # 关系id列表
                rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
    print("examples[0]", examples[0])
    print("InputExamples:", len(examples))
    return examples


def read_examples_cmim(data_dir, data_sign, rel2idx):    # jyz chg
    """load data to InputExamples
    """
    # # for test
    # if data_sign=="train":
    #     data_sign = 'val'

    examples = []

    # read src data
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['relation_list']:
                subj = triple['subject']
                obj = triple['object']
                rela = triple['predicate']
                subj_char_span = triple['subj_char_span']
                obj_char_span = triple['obj_char_span']

                if len(subj_char_span) > 1 or len(obj_char_span) > 1:
                    continue    # 删除分段实体

                subj1 = f"{subj}[span]{subj_char_span[0][0]}[span]{subj_char_span[0][1]}"
                obj1 = f"{obj}[span]{obj_char_span[0][0]}[span]{obj_char_span[0][1]}"

                en_pair_list.append([subj1, obj1])   # 实体对列表
                re_list.append(rel2idx[rela])      # 关系id列表
                rel2ens[rel2idx[rela]].append((subj1, obj1))
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
    print("examples[0]", examples[0])
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def _get_so_head(en_pair, tokenizer, text_tokens):
    """
    查找实体对在文本中的位置
    en_pair: [subj, obj]
    """
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:
        obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


def _get_so_head_cmim(en_pair,
                      span_converter: Char_Token_SpanConverter,
                      text,
                      max_text_len):    # jyz chg
    subj, subj_h, subj_t = en_pair[0].split('[span]')
    obj, obj_h, obj_t = en_pair[1].split('[span]')
    subj_tok = span_converter.tokenizer.tokenize(subj)
    obj_tok = span_converter.tokenizer.tokenize(obj)

    subj_tok_span = span_converter.get_tok_span(text, (int(subj_h), int(subj_t)))
    obj_tok_span = span_converter.get_tok_span(text, (int(obj_h), int(obj_t)))
    if subj_tok_span[-1] > max_text_len or obj_tok_span[-1] > max_text_len:  # 超出文本范围
        subj_tok_span = (-1, -1)
        obj_tok_span = (-1, -1)
    return subj_tok_span[0], obj_tok_span[0], subj_tok, obj_tok


def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
    """convert function
    """
    span_converter = Char_Token_SpanConverter(
        tokenizer, add_special_tokens=False, has_return_offsets_mapping=False)

    text_tokens = tokenizer.tokenize(example.text)
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            if '[span]' in en_pair[0]:
                sub_head, obj_head, _, _ = _get_so_head_cmim(
                    en_pair, span_converter, example.text, max_text_len)
            else:
                sub_head, obj_head, _, _ = _get_so_head(en_pair, tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                # get sub and obj head
                if '[span]' in en[0]:
                    sub_head, obj_head, sub, obj = _get_so_head_cmim(
                        en, span_converter, example.text, max_text_len)
                else:
                    sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag
            ))
        # relation judgement ablation
        if not ex_params['ensure_rel']:
            # negative samples
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag
                ))
    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            if '[span]' in en[0]:
                sub_head, obj_head, sub, obj = _get_so_head_cmim(
                    en, span_converter, example.text, max_text_len)
            else:
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length

    # features = []
    # for example in examples:
    #     feature = convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params)
    #     features.append(feature)

    # multi-process
    with Pool(1) as p:   # Pool(10)
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, ex_params=ex_params)
        features = p.map(func=convert_func, iterable=examples)

    # print(xxxxx)
    return list(chain(*features))
