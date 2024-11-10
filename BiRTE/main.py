from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
# from bert4keras.tokenizers import Tokenizer
from model import BiRTE
from util import *
from tqdm import tqdm
import random
import os
import torch.nn as nn
import torch
# from transformers.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertConfig
from transformers import BertTokenizer
import json


Print_Log_File = ""


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def judge(ex):
    '''判断样本是否正确'''
    for s, p, o in ex["triple_list"]:
        if s == '' or o == '' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


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
            return    # 跳过重复操作

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


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, args, train_data, tokenizer, predicate2id, id2predicate):
        super(data_generator, self).__init__(train_data, args.batch_size)
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.id2predicate = id2predicate

    def __iter__(self, is_random=True):
        batch_token_ids, batch_mask = [], []
        batch_s1_labels, batch_o1_labels, \
        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
        batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []

        for is_end, d in self.sample(is_random):
            if judge(d) == False:
                continue
            token_ids, _, mask = self.tokenizer.encode(
                d['text'], max_length=self.max_len
            )

            # 整理三元组 {s: [(o, p)]}
            spoes_s = {}
            spoes_o = {}
            for s, p, o in d['triple_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s_loc = (s_idx, s_idx + len(s) - 1)
                    o_loc = (o_idx, o_idx + len(o) - 1)
                    if s_loc not in spoes_s:
                        spoes_s[s_loc] = []
                    spoes_s[s_loc].append((o_loc, p))
                    if o_loc not in spoes_o:
                        spoes_o[o_loc] = []
                    spoes_o[o_loc].append((s_loc, p))
            if spoes_s and spoes_o:
                # s1_labels o1_labels
                def get_entity1_labels(item, l):
                    res = np.zeros([l, 2])
                    for start, end in item:
                        res[start][0] = 1
                        res[end][1] = 1
                    return res

                s1_labels = get_entity1_labels(spoes_s, len(token_ids))
                o1_labels = get_entity1_labels(spoes_o, len(token_ids))

                # s2_labels,o2_labels,s2_mask,o2_mask
                def get_entity2_labels_mask(item, l):
                    start, end = random.choice(list(item.keys()))
                    # 构造labels
                    labels = np.zeros((l, 2))
                    if (start, end) in item:
                        for loc, _ in item[(start, end)]:
                            labels[loc[0], 0] = 1
                            labels[loc[1], 1] = 1
                    # 构造mask
                    mask = np.zeros(l)
                    mask[start] = 1
                    mask[end] = 1
                    return labels, mask

                o2_labels, s2_mask = get_entity2_labels_mask(spoes_s, len(token_ids))
                s2_labels, o2_mask = get_entity2_labels_mask(spoes_o, len(token_ids))

                # s3_mask,o3_mask,r
                s_loc = random.choice(list(spoes_s.keys()))
                o_loc, _ = random.choice(spoes_s[s_loc])
                r = np.zeros(len(self.id2predicate))
                if s_loc in spoes_s:
                    for loc, the_r in spoes_s[s_loc]:
                        if loc == o_loc:
                            r[the_r] = 1
                s3_mask = np.zeros(len(token_ids))
                o3_mask = np.zeros(len(token_ids))
                s3_mask[s_loc[0]] = 1
                s3_mask[s_loc[1]] = 1
                o3_mask[o_loc[0]] = 1
                o3_mask[o_loc[1]] = 1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_mask.append(mask)

                batch_s1_labels.append(s1_labels)
                batch_o1_labels.append(o1_labels)

                batch_s2_mask.append(s2_mask)
                batch_o2_mask.append(o2_mask)
                batch_s2_labels.append(s2_labels)
                batch_o2_labels.append(o2_labels)

                batch_s3_mask.append(s3_mask)
                batch_o3_mask.append(o3_mask)
                batch_r.append(r)

                if len(batch_token_ids) == self.batch_size or is_end:  # 输出batch
                    batch_token_ids, batch_mask, \
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask = \
                        [sequence_padding(i).astype(np.int32)
                         for i in [batch_token_ids, batch_mask,
                                   batch_s1_labels, batch_o1_labels,
                                   batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                                   batch_s3_mask, batch_o3_mask]]

                    batch_r = np.array(batch_r).astype(np.int32)

                    yield [
                        batch_token_ids, batch_mask,
                        batch_s1_labels, batch_o1_labels,
                        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                        batch_s3_mask, batch_o3_mask, batch_r
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []


class data_generator_forCMIM2023(DataGenerator):
    """数据生成器
    """

    def __init__(self, args, train_data, tokenizer, predicate2id, id2predicate):
        super(data_generator_forCMIM2023, self).__init__(train_data, args.batch_size)
        self.max_len = args.max_len
        self.train_segment_entity_strategy = args.train_segment_entity_strategy
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.id2predicate = id2predicate
        self.span_converter = Char_Token_SpanConverter(
            tokenizer, add_special_tokens=True, has_return_offsets_mapping=False)

    def __iter__(self, is_random=True):
        batch_token_ids, batch_mask = [], []
        batch_s1_labels, batch_o1_labels, \
        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
        batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []

        for is_end, d in self.sample(is_random):
            # if judge(d)==False:   JYZ chg 2408.
            #     continue

            """ format of CMIM2023-NOM-task1-Re
            d = {
                    'id': 15, 
                    'text': '载频的ACT灯红灯闪表示信道闭塞指示。', 
                    'relation_list': [
                        {'subject': '载频', 'predicate': '含有', 'object': 'ACT灯', 'subj_char_span': [[0, 2]], 'obj_char_span': [[3, 7]]}, 
                        {'subject': '载频的ACT灯红灯闪', 'predicate': '定义', 'object': '信道闭塞指示', 'subj_char_span': [[0, 10]], 'obj_char_span': [[12, 18]]}, 
                        {'subject': '信道闭塞', 'predicate': '造成', 'object': '载频的ACT灯红灯闪', 'subj_char_span': [[12, 16]], 'obj_char_span': [[0, 10]]}
                    ]
                }
            """
            text = d['text']
            triple_list = d['relation_list']

            # token_ids, _, mask = self.tokenizer.encode(
            #     text, max_length=self.max_len
            # )
            token_info = self.tokenizer.encode_plus(
                text, max_length=self.max_len)
            token_ids = token_info['input_ids']
            mask = token_info['attention_mask']
            # print(text)
            # print(self.tokenizer.convert_ids_to_tokens(token_ids))
            # print(xxxxx)

            # 整理三元组 {s: [(o, p)]}
            spoes_s = {}
            spoes_o = {}
            # for s, p, o in d['triple_list']:
            for triple in triple_list:
                # subj = triple['subject']
                # obj = triple['object']
                rela = triple['predicate']
                rela_id = self.predicate2id[rela]

                # ----- 实体选取策略，特别是分段实体
                if self.train_segment_entity_strategy == "longest_slice":
                    # choice 1: 位置选择列表中跨度最长的那个
                    subj_char_pos = max(triple['subj_char_span'], key=lambda x: x[1] - x[0])
                    obj_char_pos = max(triple['obj_char_span'], key=lambda x: x[1] - x[0])
                elif self.train_segment_entity_strategy == "delete":
                    # choice 2: 删除所有含分段实体的三元组
                    if len(triple['subj_char_span']) > 1 or len(triple['obj_char_span']) > 1:
                        continue
                    else:
                        subj_char_pos = triple['subj_char_span'][0]
                        obj_char_pos = triple['obj_char_span'][0]
                        assert len(subj_char_pos) == 2, f"\n{subj_char_pos}"

                # 换算为token位置
                subj_tok_pos = self.span_converter.get_tok_span(text, subj_char_pos)
                obj_tok_pos = self.span_converter.get_tok_span(text, obj_char_pos)
                # print(subj)
                # print(subj_char_pos)
                # print(subj_tok_pos)
                if subj_tok_pos[1] > len(token_ids) - 1 or obj_tok_pos[1] > len(token_ids) - 1:
                    # 超出文本范围的triple不考虑
                    continue

                s_loc = (subj_tok_pos[0], subj_tok_pos[1] - 1)
                o_loc = (obj_tok_pos[0], obj_tok_pos[1] - 1)
                if s_loc not in spoes_s:
                    spoes_s[s_loc] = []
                spoes_s[s_loc].append((o_loc, rela_id))
                if o_loc not in spoes_o:
                    spoes_o[o_loc] = []
                spoes_o[o_loc].append((s_loc, rela_id))

                # s = self.tokenizer.encode(subj)[0][1:-1]   # 选择ids部分，删除[CLS][SEP]
                # p = self.predicate2id[rela]
                # o = self.tokenizer.encode(obj)[0][1:-1]
                # s_idx = search(s, token_ids)
                # o_idx = search(o, token_ids)
                # if s_idx != -1 and o_idx != -1:
                #     s_loc = (s_idx, s_idx + len(s) - 1)
                #     o_loc = (o_idx, o_idx + len(o) - 1)
                #     if s_loc not in spoes_s:
                #         spoes_s[s_loc] = []
                #     spoes_s[s_loc].append((o_loc,p))
                #     if o_loc not in spoes_o:
                #         spoes_o[o_loc] = []
                #     spoes_o[o_loc].append((s_loc,p))

            if spoes_s and spoes_o:
                """ format
                spoes_s = {
                    (subj_pos): [((obj_pos), rela_id), (...)],
                    (): [],
                    ...
                }
                """
                # s1_labels o1_labels
                def get_entity1_labels(item, l):
                    res = np.zeros([l, 2])
                    for start, end in item:
                        res[start][0] = 1
                        res[end][1] = 1
                    return res

                s1_labels = get_entity1_labels(spoes_s, len(token_ids))
                o1_labels = get_entity1_labels(spoes_o, len(token_ids))

                # s2_labels,o2_labels,s2_mask,o2_mask
                def get_entity2_labels_mask(item, l):
                    start, end = random.choice(list(item.keys()))
                    # 构造labels
                    labels = np.zeros((l, 2))
                    if (start, end) in item:
                        for loc, _ in item[(start, end)]:
                            labels[loc[0], 0] = 1
                            labels[loc[1], 1] = 1
                    # 构造mask
                    mask = np.zeros(l)
                    mask[start] = 1
                    mask[end] = 1
                    return labels, mask

                o2_labels, s2_mask = get_entity2_labels_mask(spoes_s, len(token_ids))
                s2_labels, o2_mask = get_entity2_labels_mask(spoes_o, len(token_ids))

                # s3_mask,o3_mask,r
                s_loc = random.choice(list(spoes_s.keys()))
                o_loc, _ = random.choice(spoes_s[s_loc])
                r = np.zeros(len(self.id2predicate))
                if s_loc in spoes_s:
                    for loc, the_r in spoes_s[s_loc]:
                        if loc == o_loc:
                            r[the_r] = 1
                s3_mask = np.zeros(len(token_ids))
                o3_mask = np.zeros(len(token_ids))
                s3_mask[s_loc[0]] = 1
                s3_mask[s_loc[1]] = 1
                o3_mask[o_loc[0]] = 1
                o3_mask[o_loc[1]] = 1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_mask.append(mask)

                batch_s1_labels.append(s1_labels)
                batch_o1_labels.append(o1_labels)

                batch_s2_mask.append(s2_mask)
                batch_o2_mask.append(o2_mask)
                batch_s2_labels.append(s2_labels)
                batch_o2_labels.append(o2_labels)

                batch_s3_mask.append(s3_mask)
                batch_o3_mask.append(o3_mask)
                batch_r.append(r)

                if len(batch_token_ids) == self.batch_size or is_end:  # 输出batch
                    batch_token_ids, batch_mask, \
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask = \
                        [sequence_padding(i).astype(np.int32)
                         for i in [batch_token_ids, batch_mask,
                                   batch_s1_labels, batch_o1_labels,
                                   batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                                   batch_s3_mask, batch_o3_mask]]

                    batch_r = np.array(batch_r).astype(np.int32)
                    # batch_r = np.array(batch_r, dtype=int)

                    yield [
                        batch_token_ids, batch_mask,
                        batch_s1_labels, batch_o1_labels,
                        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                        batch_s3_mask, batch_o3_mask, batch_r
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_s1_labels, batch_o1_labels, \
                        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                        batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []


class CE():
    def __call__(self, args, targets, pred, from_logist=False):
        """
        计算二分类交叉熵
        :param targets: [batch,seq,2]
        :param pred: [batch,seq,2]
        :param from_logist:是否没有经过softmax/sigmoid
        :return: loss.shape==targets.shape==pred.shape
        """
        if not from_logist:
            '''返回到没有经过softmax/sigmoid得张量'''
            # 截取pred，防止趋近于0或1,保持在[min_num,1-min_num]
            pred = torch.where(pred < 1 - args.min_num, pred, torch.ones(pred.shape).to("cuda") * 1 - args.min_num).to("cuda")
            pred = torch.where(pred > args.min_num, pred, torch.ones(pred.shape).to("cuda") * args.min_num).to("cuda")
            pred = torch.log(pred / (1 - pred))
        relu = nn.ReLU()
        # 计算传统的交叉熵loss
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        return loss


def train(args):
    global Print_Log_File

    train_path = os.path.join(args.base_path, args.dataset, "train.json")
    dev_path = os.path.join(args.base_path, args.dataset, "dev.json")
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")

    # # output_path = os.path.join(args.base_path, args.dataset, "output", args.file_id)
    output_path = args.output_dir
    # test_pred_path = os.path.join(output_path, "test_pred.json")
    # dev_pred_path = os.path.join(output_path, "dev_pred.json")
    log_path = os.path.join(output_path, "log.txt")
    Print_Log_File = os.path.join(output_path, "print_log.log")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print_config(args)

    # 加载数据集
    # train_data = json.load(open(train_path))
    # valid_data = json.load(open(dev_path))
    # test_data = json.load(open(test_path))
    # id2predicate, predicate2id = json.load(open(rel2id_path))
    with open(train_path, 'r', encoding='UTF-8') as f1:  # JYZ chg 2408
        train_data = json.loads(f1.read())
    with open(dev_path, 'r', encoding='UTF-8') as f1:
        valid_data = json.loads(f1.read())
    with open(test_path, 'r', encoding='UTF-8') as f1:
        test_data = json.loads(f1.read())
    with open(rel2id_path, 'r', encoding='UTF-8') as f1:
        id2predicate, predicate2id = json.loads(f1.read())

    # tokenizer = Tokenizer(args.bert_vocab_path)  # 注意修改
    tokenizer = BertTokenizer(args.bert_vocab_path, do_lower_case=True)  # JYZ chg 2408
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id   # 设置仅id设备可见
    # torch.cuda.set_device(int(args.cuda_id))
    # device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print_tofile(f"code will run on {args.device}", file=Print_Log_File)
    train_model = BiRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to(args.device)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataloader = data_generator(args, train_data, tokenizer, predicate2id, id2predicate)
    dataloader = data_generator_forCMIM2023(args, train_data, tokenizer, predicate2id, id2predicate)

    t_total = len(dataloader) * args.num_train_epochs

    """ 优化器准备 """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." in n],
            "weight_decay": args.weight_decay,
            "lr": args.bert_learning_rate,
        },
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." not in n],
            "weight_decay": args.weight_decay,
            "lr": args.other_learning_rate,
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    # eval & save function test
    print_tofile("function test: eval & save", file=Print_Log_File)
    output_dir = os.path.join(args.output_dir, f"checkpoint-{str(0).zfill(3)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dev_pred_path = os.path.join(output_dir, "dev_pred.json")
    dev_f1, dev_p, dev_r = evaluate_forCMIM2023(args, tokenizer, id2predicate, train_model, valid_data[:10], dev_pred_path)
    print_tofile(f"-- epoch={0}, dev , p={round(dev_p, 4)}, r={round(dev_r, 4)}, f1={round(dev_f1, 4)}",
                 file=Print_Log_File)

    best_f1 = -1.0  # 全局的best_f1
    step = 0
    binary_crossentropy = CE()
    no_change = 0
    print_tofile("start training", file=Print_Log_File)
    for epoch in range(args.num_train_epochs):
        train_model.train()
        epoch_loss = 0
        # with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
        print_tofile("", file=Print_Log_File)
        for i, batch in enumerate(dataloader):
                if i % 200 == 0:
                    print_tofile(f"    train, epoch={epoch+1}, batch={i}/{len(dataloader)}",
                                 file=Print_Log_File)
                batch = [torch.tensor(d).to("cuda") for d in batch]
                batch_token_ids, batch_mask, \
                batch_s1_labels, batch_o1_labels, \
                batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                batch_s3_mask, batch_o3_mask, batch_r = batch

                s1_pred, o1_pred, s2_pred, o2_pred, p_pred = train_model(batch_token_ids, batch_mask,
                                                                         batch_s2_mask, batch_o2_mask,
                                                                         batch_s3_mask, batch_o3_mask)

                # 计算损失
                def get_loss(target, pred, mask):
                    loss = binary_crossentropy(args, targets=target, pred=pred)  # BL2
                    loss = torch.mean(loss, dim=2).to("cuda")  # BL
                    loss = torch.sum(loss * mask).to("cuda") / torch.sum(mask).to("cuda")
                    return loss

                s1_loss = get_loss(target=batch_s1_labels, pred=s1_pred, mask=batch_mask)
                o1_loss = get_loss(target=batch_o1_labels, pred=o1_pred, mask=batch_mask)
                s2_loss = get_loss(target=batch_s2_labels, pred=s2_pred, mask=batch_mask)
                o2_loss = get_loss(target=batch_o2_labels, pred=o2_pred, mask=batch_mask)
                r_loss = binary_crossentropy(args, targets=batch_r, pred=p_pred)
                r_loss = r_loss.mean()

                loss = s1_loss + o1_loss + s2_loss + o2_loss + r_loss

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_model.zero_grad()
                # t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                # t.update(1)

        # eval & save
        epoch_loss = epoch_loss / dataloader.__len__()

        output_dir = os.path.join(args.output_dir, f"checkpoint-epoch{str(epoch+1).zfill(3)}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dev_pred_path = os.path.join(output_dir, "dev_pred.json")
        dev_f1, dev_p, dev_r = evaluate_forCMIM2023(args, tokenizer, id2predicate, train_model, valid_data, dev_pred_path)
        print_tofile(f"-- dev , epoch={epoch+1}, p={round(dev_p, 4)}, r={round(dev_r, 4)}, f1={round(dev_f1, 4)}",
                     file=Print_Log_File)

        test_pred_path = os.path.join(output_dir, "test_pred.json")
        test_f1, test_p, test_r = evaluate_forCMIM2023(args, tokenizer, id2predicate, train_model, test_data, test_pred_path)
        print_tofile(f"-- test, epoch={epoch+1}, p={round(test_p, 4)}, r={round(test_r, 4)}, f1={round(test_f1, 4)}",
                     file=Print_Log_File)

        # if f1 > best_f1:
        #     # Save model checkpoint
        #     best_f1 = f1
        #     torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))  # 保存最优模型权重

        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f" % (
                int(epoch), epoch_loss, dev_f1, dev_p, dev_r, 0.0), file=f)

    # # 对test集合进行预测
    # # 加载训练好的权重
    # train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    # f1, precision, recall = evaluate(args, tokenizer, id2predicate, train_model, test_data, test_pred_path)
    # with open(log_path, "a", encoding="utf-8") as f:
    #     print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)


def extract_spoes(args, tokenizer, id2predicate, model, text, entity_start=0.5, entity_end=0.5, p_num=0.5):
    """抽取输入text所包含的三元组
    """
    # sigmoid=nn.Sigmoid()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # model.to("cuda")
    model.to(args.device)
    tokens = tokenizer.tokenize(text, max_length=args.max_len)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, _, mask = tokenizer.encode(text, max_length=args.max_len)
    # 获取BERT表示
    model.eval()
    with torch.no_grad():
        head, tail, rel, cls = model.get_embed(torch.tensor([token_ids]).to("cuda"), torch.tensor([mask]).to("cuda"))
        head = head.cpu().detach().numpy()  # [1,L,H]
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()

    def get_entity(entity_pred):
        start = np.where(entity_pred[0, :, 0] > entity_start)[0]
        end = np.where(entity_pred[0, :, 1] > entity_end)[0]
        entity = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity

    # 抽取s1 o1
    model.eval()
    with torch.no_grad():
        s1_preds = model.s_pred(torch.tensor(head).to("cuda"), torch.tensor(cls).to("cuda"))
        o1_preds = model.o_pred(torch.tensor(tail).to("cuda"), torch.tensor(cls).to("cuda"))

        s1_preds = s1_preds.cpu().detach().numpy()  # [1,L,2]
        o1_preds = o1_preds.cpu().detach().numpy()  # [1,L,2]

        s1_preds[:, 0, :], s1_preds[:, -1, :] = 0.0, 0.0
        o1_preds[:, 0, :], o1_preds[:, -1, :] = 0.0, 0.0

    s1 = get_entity(s1_preds)
    o1 = get_entity(o1_preds)

    # 获得s_loc,o_loc
    pairs_0 = []
    for s in s1:
        for o in o1:
            pairs_0.append((s[0], s[1], o[0], o[1]))

    pairs_1 = []
    for s in s1:
        # s:(start,end)
        s2_mask = np.zeros(len(token_ids)).astype(np.int)
        s2_mask[s[0]] = 1
        s2_mask[s[1]] = 1

        model.eval()
        with torch.no_grad():
            o2_pred = model.o_pred_from_s(torch.tensor(head).to("cuda"), torch.tensor(tail).to("cuda"),
                                          torch.tensor([s2_mask]).to("cuda"), cls=torch.tensor(cls).to("cuda"))
            o2_pred = o2_pred.cpu().detach().numpy()  # [1,L,2]
            o2_pred[:, 0, :], o2_pred[:, -1, :] = 0.0, 0.0
        objects2 = get_entity(o2_pred)
        if objects2:
            for o in objects2:
                pairs_1.append((s[0], s[1], o[0], o[1]))

    pairs_2 = []
    for o in o1:
        # o:(start,end)
        o2_mask = np.zeros(len(token_ids)).astype(np.int)
        o2_mask[o[0]] = 1
        o2_mask[o[1]] = 1

        model.eval()
        with torch.no_grad():
            s2_pred = model.s_pred_from_o(torch.tensor(head).to("cuda"), torch.tensor(tail).to("cuda"),
                                          torch.tensor([o2_mask]).to("cuda"), cls=torch.tensor(cls).to("cuda"))
            s2_pred = s2_pred.cpu().detach().numpy()  # [1,L,2]
            s2_pred[:, 0, :], s2_pred[:, -1, :] = 0.0, 0.0
        subjects2 = get_entity(s2_pred)
        if subjects2:
            for s in subjects2:
                pairs_2.append((s[0], s[1], o[0], o[1]))

    pairs_1 = set(pairs_1)
    pairs_2 = set(pairs_2)

    pairs = list(pairs_1 | pairs_2)

    if pairs:  # m * 4
        s_mask = np.zeros([len(pairs), len(token_ids)]).astype(np.int)
        o_mask = np.zeros([len(pairs), len(token_ids)]).astype(np.int)

        for i, pair in enumerate(pairs):
            s1, s2, o1, o2 = pair
            s_mask[i, s1] = 1
            s_mask[i, s2] = 1
            o_mask[i, o1] = 1
            o_mask[i, o2] = 1

        spoes = []
        rel = np.repeat(rel, len(pairs), 0)

        # 传入subject，抽取object和predicate
        model.eval()
        with torch.no_grad():
            p_pred = model.p_pred(
                rel=torch.tensor(rel).to("cuda"),
                s_mask=torch.tensor(s_mask).to("cuda"),
                o_mask=torch.tensor(o_mask).to("cuda"),
            )
            p_pred = p_pred.cpu().detach().numpy()  # BR

        index, p_index = np.where(p_pred > p_num)
        for i, p in zip(index, p_index):
            s1, s2, o1, o2 = pairs[i]
            spoes.append(
                (
                    (mapping[s1][0], mapping[s2][-1]),
                    p,
                    (mapping[o1][0], mapping[o2][-1])
                )
            )

        return [(text[s[0]:s[1] + 1], id2predicate[str(p)], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


def extract_spoes_forCMIM2023(args, tokenizer, id2predicate, model, text,
                              entity_start=0.5, entity_end=0.5, p_num=0.5):
    """抽取输入text所包含的三元组
    """
    # sigmoid=nn.Sigmoid()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(args.device)

    # tokens = tokenizer.tokenize(text, max_length=args.max_len)
    # mapping = tokenizer.rematch(text, tokens)  # mapping的下标是token下标，映射到char下标
    span_converter = Char_Token_SpanConverter(
        tokenizer, add_special_tokens=True, has_return_offsets_mapping=False)
    mapping = span_converter.get_mapping_tok2char(text)

    # token_ids, _, mask = tokenizer.encode(text, max_length=args.max_len)
    token_info = tokenizer.encode_plus(
        text, max_length=args.max_len)
    token_ids = token_info['input_ids']
    mask = token_info['attention_mask']

    # 获取BERT表示
    model.eval()
    with torch.no_grad():
        head, tail, rel, cls = model.get_embed(torch.tensor([token_ids]).to(args.device), torch.tensor([mask]).to(args.device))
        head = head.cpu().detach().numpy()  # [1,L,H]
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()

    def get_entity(entity_pred):
        start = np.where(entity_pred[0, :, 0] > entity_start)[0]
        end = np.where(entity_pred[0, :, 1] > entity_end)[0]
        entity = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity

    # 抽取s1 o1
    model.eval()
    with torch.no_grad():
        s1_preds = model.s_pred(torch.tensor(head).to(args.device), torch.tensor(cls).to(args.device))
        o1_preds = model.o_pred(torch.tensor(tail).to(args.device), torch.tensor(cls).to(args.device))

        s1_preds = s1_preds.cpu().detach().numpy()  # [1,L,2]
        o1_preds = o1_preds.cpu().detach().numpy()  # [1,L,2]

        s1_preds[:, 0, :], s1_preds[:, -1, :] = 0.0, 0.0
        o1_preds[:, 0, :], o1_preds[:, -1, :] = 0.0, 0.0

    s1 = get_entity(s1_preds)
    o1 = get_entity(o1_preds)

    # 获得s_loc,o_loc
    pairs_0 = []
    for s in s1:
        for o in o1:
            pairs_0.append((s[0], s[1], o[0], o[1]))

    pairs_1 = []
    for s in s1:
        # s:(start,end)
        s2_mask = np.zeros(len(token_ids), dtype=int)
        s2_mask[s[0]] = 1
        s2_mask[s[1]] = 1

        model.eval()
        with torch.no_grad():
            o2_pred = model.o_pred_from_s(torch.tensor(head).to(args.device), torch.tensor(tail).to(args.device),
                                          torch.tensor([s2_mask]).to(args.device), cls=torch.tensor(cls).to(args.device))
            o2_pred = o2_pred.cpu().detach().numpy()  # [1,L,2]
            o2_pred[:, 0, :], o2_pred[:, -1, :] = 0.0, 0.0
        objects2 = get_entity(o2_pred)
        if objects2:
            for o in objects2:
                pairs_1.append((s[0], s[1], o[0], o[1]))

    pairs_2 = []
    for o in o1:
        # o:(start,end)
        o2_mask = np.zeros(len(token_ids), dtype=int)
        o2_mask[o[0]] = 1
        o2_mask[o[1]] = 1

        model.eval()
        with torch.no_grad():
            s2_pred = model.s_pred_from_o(torch.tensor(head).to(args.device), torch.tensor(tail).to(args.device),
                                          torch.tensor([o2_mask]).to(args.device), cls=torch.tensor(cls).to(args.device))
            s2_pred = s2_pred.cpu().detach().numpy()  # [1,L,2]
            s2_pred[:, 0, :], s2_pred[:, -1, :] = 0.0, 0.0
        subjects2 = get_entity(s2_pred)
        if subjects2:
            for s in subjects2:
                pairs_2.append((s[0], s[1], o[0], o[1]))

    pairs_1 = set(pairs_1)
    pairs_2 = set(pairs_2)

    pairs = list(pairs_1 | pairs_2)

    if pairs:  # m * 4
        s_mask = np.zeros([len(pairs), len(token_ids)], dtype=int)
        o_mask = np.zeros([len(pairs), len(token_ids)], dtype=int)

        for i, pair in enumerate(pairs):
            s1, s2, o1, o2 = pair
            s_mask[i, s1] = 1
            s_mask[i, s2] = 1
            o_mask[i, o1] = 1
            o_mask[i, o2] = 1

        spoes = []
        rel = np.repeat(rel, len(pairs), 0)

        # 传入subject，抽取object和predicate
        model.eval()
        with torch.no_grad():
            p_pred = model.p_pred(
                rel=torch.tensor(rel).to(args.device),
                s_mask=torch.tensor(s_mask).to(args.device),
                o_mask=torch.tensor(o_mask).to(args.device),
            )
            p_pred = p_pred.cpu().detach().numpy()  # BR

        index, p_index = np.where(p_pred > p_num)
        for i, p in zip(index, p_index):
            s1, s2, o1, o2 = pairs[i]
            spoes.append(
                (
                    (mapping[s1][0], mapping[s2][-1]),
                    p,
                    (mapping[o1][0], mapping[o2][-1])
                )
            )

        # return [(text[s[0]:s[1] + 1], id2predicate[str(p)], text[o[0]:o[1] + 1])
        #         for s, p, o, in spoes]
        return [(text[s[0]:s[1]], id2predicate[str(p)], text[o[0]:o[1]])
                for s, p, o, in spoes]
    else:
        return []


def evaluate(args, tokenizer, id2predicate, model, evl_data, pred_output_file):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(pred_output_file, 'w', encoding='utf-8')
    pbar = tqdm()
    for d in evl_data:
        R = set(extract_spoes(args, tokenizer, id2predicate, model, d['text']))
        T = set([(i[0], i[1], i[2]) for i in d['triple_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'triple_list': list(T),
            'triple_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }, ensure_ascii=False, indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


def evaluate_forCMIM2023(args, tokenizer, id2predicate, model, evl_data, pred_output_file):
    """评估函数，计算f1、precision、recall
    """
    correct_num, pred_num, label_num = 1e-10, 1e-10, 1e-10
    # f = open(pred_output_file, 'w', encoding='utf-8')
    sample_triples_pred = []   # 存放抽取的三元组结果

    # pbar = tqdm()
    for d in evl_data:
        triples_pred = set(extract_spoes_forCMIM2023(args, tokenizer, id2predicate, model, d['text'])) # !!!!!!!!!
        triples_label = set([(triple['subject'], triple['predicate'], triple['object'])
                 for triple in d['relation_list']])
        correct_num += len(triples_pred & triples_label)
        pred_num += len(triples_pred)
        label_num += len(triples_label)
        f1, precision, recall = 2 * correct_num / (pred_num + label_num), correct_num / pred_num, correct_num / label_num
        # pbar.update()
        # pbar.set_description(
        #     'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        # )

        sample_triples_pred.append({
            'text': d['text'],
            'triple_list_label': list(triples_label),
            'triple_list_pred': list(triples_pred),
        })
        # s = json.dumps({
        #     'text': d['text'],
        #     'triple_list': list(triples_label),
        #     'triple_list_pred': list(triples_pred),
        #     'new': list(triples_pred - triples_label),
        #     'lack': list(triples_label - triples_pred),
        # }, ensure_ascii=False, indent=4)
        # f.write(s + '\n')
    # pbar.close()
    # f.close()
    with open(pred_output_file, "w", encoding="utf-8") as file1:
        json.dump(sample_triples_pred, file1, ensure_ascii=False, indent=4)

    return f1, precision, recall


def test(args):
    torch.cuda.set_device(int(args.cuda_id))
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    output_path = os.path.join(args.base_path, args.dataset, "output", args.file_id)
    test_pred_path = os.path.join(output_path, "test_pred.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    config = BertConfig.from_pretrained(args.bert_config_path)
    # tokenizer = Tokenizer(args.bert_vocab_path)
    tokenizer = BertTokenizer(args.bert_vocab_path)
    config.num_p = len(id2predicate)
    train_model = BiRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args, tokenizer, id2predicate, train_model, test_data, test_pred_path)
    print("f1:%f, precision:%f, recall:%f" % (f1, precision, recall))
