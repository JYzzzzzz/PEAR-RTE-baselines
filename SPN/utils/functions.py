import torch, collections
import json


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


def list_index(list1: list, list2: list) -> list:
    # 查找 list1 在 list2 中的位置
    start = [i for i, x in enumerate(list2) if x == list1[0]]
    end = [i for i, x in enumerate(list2) if x == list1[-1]]
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]
    else:
        for i in start:
            for j in end:
                if i <= j:
                    if list2[i:j+1] == list1:
                        index = (i, j)
                        break
        return index[0], index[1]





# def list_index(list1: list, list2: list) -> list:
#     start = [i for i, x in enumerate(list2) if x == list1[0]]
#     end = [i for i, x in enumerate(list2) if x == list1[-1]]
#     if len(start) == 1 and len(end) == 1:
#         return start[0], end[0]
#     else:
#         for i in start:
#             for j in end:
#                 if i <= j:
#                     if list2[i:j+1] == list1:
#                         break
#         return i, j
#

def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
    "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
    "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


def data_process(input_doc, relational_alphabet, tokenizer, args):
    samples = []
    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
        """
        {
            "sentText": "Hockenheim , Germany .", 
            "relationMentions": [{"em1Text": "Germany", "em2Text": "Hockenheim", "label": "/location/location/contains"}]
        }
        """

    for i in range(len(lines)):
        token_sent = [tokenizer.cls_token] + \
                     tokenizer.tokenize(remove_accents(lines[i]["sentText"])) + \
                     [tokenizer.sep_token]
        #   # remove_accents 将一些特殊符号转换为常见符号
        triples = lines[i]["relationMentions"]

        target = {"relation": [],
                  "head_start_index": [], "head_end_index": [],
                  "tail_start_index": [], "tail_end_index": []}
        for triple in triples:
            head_entity = remove_accents(triple["em1Text"])
            tail_entity = remove_accents(triple["em2Text"])
            head_token = tokenizer.tokenize(head_entity)
            tail_token = tokenizer.tokenize(tail_entity)
            relation_id = relational_alphabet.get_index(triple["label"])

            # 查找实体token在句子token中的位置
            head_start_index, head_end_index = list_index(head_token, token_sent)
            assert head_end_index >= head_start_index
            tail_start_index, tail_end_index = list_index(tail_token, token_sent)
            assert tail_end_index >= tail_start_index

            target["relation"].append(relation_id)
            target["head_start_index"].append(head_start_index)
            target["head_end_index"].append(head_end_index)
            target["tail_start_index"].append(tail_start_index)
            target["tail_end_index"].append(tail_end_index)
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])
    return samples


def data_process_2407(input_doc, relational_alphabet, tokenizer, args):
    # 为了对接本人研究内容的中文数据集，基于data_process()进行修改

    samples = []
    with open(input_doc, 'r', encoding='UTF-8') as f:
        lines = json.loads(f.read())

    char_token_converter = Char_Token_SpanConverter(
        tokenizer, add_special_tokens=True, has_return_offsets_mapping=False)

    for i in range(len(lines)):
        token_sent = [tokenizer.cls_token] + \
                     tokenizer.tokenize(remove_accents(lines[i]["text"])) + \
                     [tokenizer.sep_token]
        #   # remove_accents 将一些特殊符号转换为常见符号
        if len(token_sent) > args.max_text_token_len:  # 句子长度限制
            token_sent = token_sent[:args.max_text_token_len-1] + [tokenizer.sep_token]

        triples = lines[i]["relation_list"]

        target = {"relation": [],
                  "head_start_index": [], "head_end_index": [],
                  "tail_start_index": [], "tail_end_index": []}
        for triple in triples:
            if len(triple['subj_char_span']) > 1 or len(triple['obj_char_span']) > 1:
                continue  # delete all triples with segmented entities

            head_entity = remove_accents(triple["subject"])
            tail_entity = remove_accents(triple["object"])
            head_token = tokenizer.tokenize(head_entity)
            tail_token = tokenizer.tokenize(tail_entity)
            relation_id = relational_alphabet.get_index(triple["predicate"])
            subj_char_span = triple["subj_char_span"][0]
            obj_char_span = triple["obj_char_span"][0]

            # 查找实体token在句子token中的位置
            # head_start_index, head_end_index = list_index(head_token, token_sent)
            # assert head_end_index >= head_start_index
            # tail_start_index, tail_end_index = list_index(tail_token, token_sent)
            # assert tail_end_index >= tail_start_index
            head_span = char_token_converter.get_tok_span(
                remove_accents(lines[i]["text"]), subj_char_span)
            head_start_index = head_span[0]
            head_end_index = head_span[1] - 1
            tail_span = char_token_converter.get_tok_span(
                remove_accents(lines[i]["text"]), obj_char_span)
            tail_start_index = tail_span[0]
            tail_end_index = tail_span[1] - 1
            if head_span[1] > args.max_text_token_len-1 or tail_span[1] > args.max_text_token_len-1:
                continue   # 在句子长度限制之外的三元组删除

            target["relation"].append(relation_id)
            target["head_start_index"].append(head_start_index)
            target["head_end_index"].append(head_end_index)
            target["tail_start_index"].append(tail_start_index)
            target["tail_end_index"].append(tail_end_index)
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])

    return samples


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, info, args):
    seq_lens = info["seq_len"] # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx) in zip(start_probs, end_probs, seq_lens, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], args.n_best_size)
            end_indexes = _get_best_indexes(end_prob[triple_id], args.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len-1): # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_span_length:
                        continue
                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                        )
                    )
            output[sent_idx][triple_id] = predictions
    return output


def generate_relation(pred_rel_logits, info, args):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            output[sent_idx][triple_id] = _Prediction(
                            pred_rel=pred_rel[triple_id],
                            rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob"]
    )
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], info, args)
    pred_tail_ent_dict = generate_span(output["tail_start_logits"], output["tail_end_logits"], info, args)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args)
    triples = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            pred_head = pred_head_ent_dict[sent_idx][triple_id]
            pred_tail = pred_tail_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple)
    # print(triples)
    return triples


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head and pred_tail:
            for ele in pred_head:
                if ele.start_index != 0:
                    break
            head = ele
            for ele in pred_tail:
                if ele.start_index != 0:
                    break
            tail = ele
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index, head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob, tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob, tail_end_prob=tail.end_prob)
        else:
            return
    else:
        return


# def strict_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
#     if pred_rel.pred_rel != num_classes:
#         if pred_head and pred_tail:
#             if pred_head[0].start_index != 0 and pred_tail[0].start_index != 0:
#                 return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=pred_head[0].start_index, head_end_index=pred_head[0].end_index, head_start_prob=pred_head[0].start_prob, head_end_prob=pred_head[0].end_prob, tail_start_index=pred_tail[0].start_index, tail_end_index=pred_tail[0].end_index, tail_start_prob=pred_tail[0].start_prob, tail_end_prob=pred_tail[0].end_prob)
#             else:
#                 return
#         else:
#             return
#     else:
#         return


def formulate_gold(target, info):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            gold[sent_idxes[i]].append(
                (target[i]["relation"][j].item(), target[i]["head_start_index"][j].item(), target[i]["head_end_index"][j].item(), target[i]["tail_start_index"][j].item(), target[i]["tail_end_index"][j].item())
            )
    return gold


