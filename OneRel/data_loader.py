from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
import numpy as np
from random import choice
from transformers import BertTokenizer

# from utils import get_tokenizer
# from utils.tokenization import BasicTokenizer
# tokenizer = get_tokenizer('pre_trained_bert/vocab.txt')
# basicTokenizer = BasicTokenizer(do_lower_case=False)
# tag_file = 'data/tag.txt'

class ZhTokenizer:
    def __init__(self, model_path='pre_trained_bert/chinese-bert-wwm-ext/vocab.txt'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.vocab2id = self.tokenizer.vocab

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return_tokens = ["[CLS]"]
        for token in tokens:
            return_tokens.append(token)
            return_tokens.append("[unused1]")
        return_tokens += ["[SEP]"]
        return return_tokens

    def encode(self, text):
        return_tokens = self.tokenize(text)
        input_ids = [int(self.vocab2id.get(token, 100)) for token in return_tokens]
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

    def get_onerel_token_pos(self, normal_token_pos: int):
        """
        normal_token_pos: self.tokenizer.tokenize(text) 生成的tokens中的某个位置
        return: self.tokenize(text) 生成的tokens中的对应位置
        """
        return normal_token_pos*2+1


# class Char_Token_SpanConverter(object):  # jyz add 2024-06. 用于中文数据集生成准确的token_span
#     # version: 240710
#     def __init__(self, tokenizer, add_special_tokens=False, has_return_offsets_mapping=True):
#         """
#         add_special_tokens: 如果 add_special_tokens=True，会将 [CLS] 考虑在内，token_span 数值整体+1
#         has_return_offsets_mapping: bool. tokenizer是否包含return_offsets_mapping功能，若不包含，手动生成。
#         """
#         self.tokenizer = tokenizer
#         self.token_info = None
#         self.error_tok_spans = []  # {text, char_span, char_span_str, tok_span_str}
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
#         # # check
#         # self._char_tok_span_check(char_span, token_span)
#         return tuple(token_span)
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
#             # Tokenizer 自带生成 offset_mapping(tok2char_mapping) 的功能
#             token_info = self.tokenizer.encode_plus(text,
#                                                     return_offsets_mapping=True,
#                                                     add_special_tokens=self.add_special_tokens)
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
#                                                     add_special_tokens=self.add_special_tokens)
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
#                 c_belong_unk = 0
#                 c_tokens = self.tokenizer.tokenize(c)
#                 if len(c_tokens) == 0:  # c 是一个空白字符
#                     pass
#                 else:
#                     ct = c_tokens[0]
#                     # 查找字符在哪个token中
#                     while ct not in tokens[tokens_i[0]]:
#                         if tokens[tokens_i[0]] == '[UNK]' and ct not in tokens[tokens_i[0]+1]:
#                             c_belong_unk = 1
#                             break
#                         tokens_i[0] += 1
#                         tokens_i[1] = 0
#                         assert tokens_i[0] < len(tokens), f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
#                     if ct == '[UNK]':
#                         c_belong_unk = 1
#
#                     if c_belong_unk == 0:
#                         # 查找字符在token中哪个位置
#                         ct_pos = tokens[tokens_i[0]].find(ct, tokens_i[1])
#                         assert ct_pos >= tokens_i[1], f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
#                         # 添加到char2tok_mapping
#                         char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
#                         # 更新tokens_i
#                         tokens_i[1] = ct_pos + len(ct)
#                         if tokens_i[1] >= len(tokens[tokens_i[0]]):
#                             tokens_i[0] += 1
#                             tokens_i[1] = 0
#                     else:
#                         char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
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
#         sub_text_from_char = self.tokenizer.decode(self.tokenizer.encode(sub_text_from_char0, add_special_tokens=False))
#
#         sub_text_from_token = self.tokenizer.decode(self.token_info['input_ids'][tok_span[0]:tok_span[1]])
#
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


def find_head_idx(source, target):  # 自动查找实体位置
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class REDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer_for_onerel):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.onerel_tokenizer = tokenizer_for_onerel  # 在原始生成的每个token后新增一个[unused1]

        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:2]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        if self.is_test:
            self.json_data = self.json_data[:1000]  # self.json_data 为数据集内容
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))  # jyz chg
        self.tag2id = json.load(open('data/tag2id.json'))[1]

        self.char_token_spanconverter = Char_Token_SpanConverter(
            self.onerel_tokenizer.tokenizer, add_special_tokens=False,
            has_return_offsets_mapping=False)
        self.tokenize_examples = []
        self.text_examples = []

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]   # 数据集一个sample

        text = ins_json_data['text']
        # text = ' '.join(text.split()[:self.config.max_len])
        # text = basicTokenizer.tokenize(text)
        # text = " ".join(text[:self.config.max_len])
        onerel_tokens = self.onerel_tokenizer.tokenize(text)  # 夹着 [unused1]
        if len(onerel_tokens) > self.config.bert_max_len:
            onerel_tokens = onerel_tokens[: self.config.bert_max_len]
        text_len = len(onerel_tokens)
        # 显示几个示例
        if len(self.text_examples) < 3:
            print(f"-------------------- REDataset __getitem__() examples")
            print(f"text = {text}")
            print(f"tokens = {onerel_tokens}")
            self.text_examples.append(text)

        if not self.is_test:
            s2ro_map = {}
            # print(ins_json_data)
            for triple_info in ins_json_data['relation_list']:
                """
                triple is this format in my dataset: 
                {
                    "subject": "SAE-GW",
                    "predicate": "组成部分",
                    "object": "ServingGW",
                    "subj_char_span": [[18, 24]],
                    "obj_char_span": [[0, 9]],
                },
                """

                # jyz chg 2409. 删除所有含分段实体的三元组
                if len(triple_info['subj_char_span']) > 1 or len(triple_info['obj_char_span']) > 1:
                    continue  # delete all triples with segmented entities
                subj_char_span = triple_info['subj_char_span'][0]
                obj_char_span = triple_info['obj_char_span'][0]
                assert len(subj_char_span) == 2, f"\n{ins_json_data}\n{triple_info}"

                # jyz chg 2406. 获取三元组文本
                # triple = (self.tokenizer.tokenize(triple[0])[1:-1],
                #           triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                triple = (self.onerel_tokenizer.tokenize(triple_info['subject'])[1:-1],
                          triple_info['predicate'],
                          self.onerel_tokenizer.tokenize(triple_info['object'])[1:-1])

                # jyz chg 2406. 获取三元组token位置
                # sub_head_idx = find_head_idx(tokens, triple[0])
                # obj_head_idx = find_head_idx(tokens, triple[2])
                # if sub_head_idx != -1 and obj_head_idx != -1:
                #     subj_tok_span = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                #     if subj_tok_span not in s2ro_map:
                #         s2ro_map[subj_tok_span] = []
                #     s2ro_map[subj_tok_span].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
                subj_tok_span = self.char_token_spanconverter.get_tok_span(
                    text, subj_char_span)
                obj_tok_span = self.char_token_spanconverter.get_tok_span(
                    text, obj_char_span)
                sub_head_idx = self.onerel_tokenizer.get_onerel_token_pos(subj_tok_span[0])
                obj_head_idx = self.onerel_tokenizer.get_onerel_token_pos(obj_tok_span[0])
                sub_tail_idx = sub_head_idx + len(triple[0]) - 1
                obj_tail_idx = obj_head_idx + len(triple[2]) - 1
                if sub_tail_idx < self.config.bert_max_len and obj_tail_idx < self.config.bert_max_len:
                    subj_tok_span = (sub_head_idx, sub_tail_idx)
                    if subj_tok_span not in s2ro_map:
                        s2ro_map[subj_tok_span] = []
                    s2ro_map[subj_tok_span].append((obj_head_idx, obj_tail_idx, self.rel2id[triple[1]]))

                # 显示示例
                if len(self.tokenize_examples) < 3:
                    subj_tokens = onerel_tokens[sub_head_idx:sub_head_idx + len(triple[0])]
                    obj_tokens = onerel_tokens[obj_head_idx:obj_head_idx + len(triple[2])]
                    str_temp = f"triple_label_token is {triple}, " \
                               f"\ntriple_label_token check is {(subj_tokens, triple_info['predicate'], obj_tokens)}"
                    print(str_temp)
                    self.tokenize_examples.append(str_temp)
            # ^^^ for triple_info in ins_json_data['relation_list']:

            if s2ro_map:
                token_ids, segment_ids = self.onerel_tokenizer.encode(text)
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                mask_length = len(masks)
                token_ids = np.array(token_ids)
                # masks = np.array(masks) + 1
                masks = np.array(masks)
                loss_masks = np.ones((mask_length, mask_length))
                triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))
                for s in s2ro_map:
                    sub_head = s[0]
                    sub_tail = s[1]
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_head, obj_tail, relation = ro
                        triple_matrix[relation][sub_head][obj_head] = self.tag2id['HB-TB']
                        triple_matrix[relation][sub_head][obj_tail] = self.tag2id['HB-TE']
                        triple_matrix[relation][sub_tail][obj_tail] = self.tag2id['HE-TE']

                return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['relation_list'], onerel_tokens
            else:
                # print("无三元组样本", ins_json_data)
                return None

        else:   # self.is_test is True
            token_ids, masks = self.onerel_tokenizer.encode(text)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            # masks = np.array(masks) + 1
            masks = np.array(masks)
            mask_length = len(masks)
            # loss_masks = np.array(masks) + 1
            loss_masks = np.array(masks)
            triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))
            return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['relation_list'], onerel_tokens


def re_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[3], reverse=True)
    if not batch:
        print("  a batch is <None> in re_collate_fn(batch)")
        return None
    for sample in batch:
        if len(sample) < 7:
            print(f"  a sample does not contain enough values{len(sample)} in re_collate_fn(batch)")
            print(f"  batch len = {len(batch)}")
            print(sample)

    token_ids, masks, loss_masks, text_len, triple_matrix, triples, tokens = zip(*batch)
    cur_batch_len = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_loss_masks = torch.LongTensor(cur_batch_len, 1, max_text_len, max_text_len).zero_()

    # if use WebNLG_star, modify 24 to 171
    # if use duie, modify 24 to 48
    # batch_triple_matrix = torch.LongTensor(cur_batch_len, 24, max_text_len, max_text_len).zero_()
    batch_triple_matrix = torch.LongTensor(cur_batch_len, 13, max_text_len, max_text_len).zero_()

    for i in range(cur_batch_len):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_loss_masks[i, 0, :text_len[i], :text_len[i]].copy_(torch.from_numpy(loss_masks[i]))
        batch_triple_matrix[i, :, :text_len[i], :text_len[i]].copy_(torch.from_numpy(triple_matrix[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'loss_mask': batch_loss_masks,
            'triple_matrix': batch_triple_matrix,
            'triples': triples,
            'tokens': tokens}


def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=re_collate_fn):
    tokenizer = ZhTokenizer(model_path=config.pretrain_model_path)
    dataset = REDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        if self.next_data is None:  # jyz chg 2024-07
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

