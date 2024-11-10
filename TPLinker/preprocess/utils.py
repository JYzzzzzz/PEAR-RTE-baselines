import re
from tqdm import tqdm
# from IPython.core.debugger import set_trace
import copy
import json
import os


class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, hyperparameter):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.log("============================================================================")
        self.log("project: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        hyperparameters_format = "--------------hypter_parameters------------------- \n{}\n-----------------------------------------"
        self.log(hyperparameters_format.format(json.dumps(hyperparameter, indent=4)))

    def log(self, text):
        text = "run_id: {}, {}".format(self.run_id, text)
        print(text)
        open(self.log_path, "a", encoding="utf-8").write("{}\n".format(text))


class Preprocessor:
    '''
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the relations, which will be used in tagging phase
    '''

    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def transform_data(self, data, ori_format, dataset_type, add_id=True):
        '''
        This function can only deal with three original format used in the previous works. 
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "joint_re", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc="Transforming data format"):
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key],
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            normal_sample_list.append(normal_sample)

        return self._clean_sp_char(normal_sample_list)

    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len=50, encoder="BERT", data_type="train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc="Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)  # 返回句子中单词的偏移量

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):  # sliding_len 滑动窗口的大小，
                if encoder == "BERT":  # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len  # 结束的长度

                char_span_list = tok2char_span[start_ind:end_ind]  # 截断
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]  # 第一个词到最后一个词的长度
                sub_text = text[char_level_span[0]:char_level_span[1]]  # 原始文本截断

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,  # token的偏移量
                    "char_offset": char_level_span[0],  # 每个字符的偏移量
                }
                if data_type == "test":  # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:  # train or valid dataset, only save spo and entities in the subtext
                    # spo
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are both in this subtext, add this spo to new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                                and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind]  # start_ind: tok level offset
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0]  # char level offset
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)

                    # entity
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # if entity in this subtext, add the entity to new sample
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind:
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]

                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)

                    # event
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            new_event = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in new_event["argument_list"]:
                                if arg["tok_span"][0] >= start_ind and arg["tok_span"][1] <= end_ind:
                                    new_arg_list.append(arg)
                            new_event["argument_list"] = new_arg_list
                            sub_event_list.append(new_event)
                        new_sample["event_list"] = sub_event_list  # maybe empty

                    new_sample["entity_list"] = sub_ent_list  # maybe empty
                    new_sample["relation_list"] = sub_rel_list  # maybe empty
                    split_sample_list.append(new_sample)

                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break

            new_sample_list.extend(split_sample_list)
        return new_sample_list

    def _clean_sp_char(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
            #             text = re.sub("([A-Za-z]+)", r" \1 ", text)
            #             text = re.sub("(\d+)", r" \1 ", text)
            #             text = re.sub("\s+", " ", text).strip()
            return text

        for sample in tqdm(dataset, desc="Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset

    def clean_data_wo_span(self, ori_data, separate=False, data_type="train"):
        """
        rm duplicate whitespaces
        and add whitespaces around tokens to keep special characters from them
        :param: separate: 如果separate参数为真（True），则执行以下操作：
                            使用正则表达式将非字母数字字符前后添加空格。
                            再次使用正则表达式将连续的空白字符替换为单个空格，并去除首尾的空白字符。
        """

        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub("\s+", " ", text).strip()
            return text

        def clean_text_and_generate_mapping(text):
            # JYZ add 2024-06-04. 仅适用于中文

            cleaned_text = re.sub("\s+", " ", text).strip()
            ##### 使用正则表达式将连续的空白字符替换为单个空格，并去除首尾的空白字符
            ##### 空白字符为：（空格）（制表符）（换行符）（回车符）（垂直制表符）（换页符）

            mapping = [-1] * len(text)
            text_i = 0
            cleaned_text_i = 0
            while text_i < len(text) and cleaned_text_i < len(cleaned_text):
                if bool(re.match("\s", text[text_i])) is False:
                    if text[text_i] == cleaned_text[cleaned_text_i]:
                        mapping[text_i] = cleaned_text_i
                        text_i += 1
                        cleaned_text_i += 1
                    else:
                        assert 0, f"\n{text}\n{cleaned_text}" \
                                  f"\n{text_i}  {text[text_i]}" \
                                  f"\n{cleaned_text_i}  {cleaned_text[cleaned_text_i]}"
                else:
                    if bool(re.match("\s", cleaned_text[cleaned_text_i])):
                        cleaned_text_i += 1
                    mapping[text_i] = -1    # text中所有空白字符对应 -1 位置
                    text_i += 1

            if cleaned_text_i < len(cleaned_text):
                assert 0, f"\n{text}\n{cleaned_text}\n[{cleaned_text[cleaned_text_i:]}]"
            if text_i < len(text):
                for c in text[text_i:]:
                    assert bool(re.match("\s", c)), f"\n{text}\n{cleaned_text}\n[{text[text_i:]}]"

            return cleaned_text, mapping

        def text_clean_pos_mapping(span: list, themap: list):
            """
            位置映射。
            span为[x,y]，要求x<y，要符合切片规则.
            mapping 为 clean_text_and_generate_mapping 生成的mapping
            return 与span规则相同
            """
            assert len(span) == 2, f"\n{span}"

            offset = 0
            span_map_l = themap[span[0]]
            while span_map_l == -1:
                offset += 1
                span_map_l = themap[span[0]+offset]

            offset = 0
            span_map_r = themap[span[1]-1]
            while span_map_r == -1:
                offset += 1
                span_map_r = themap[span[1] - 1-offset]
            span_map_r += 1

            return [span_map_l, span_map_r]

        for sample in tqdm(ori_data, desc="clean data"):
            # sample["text"] = clean_text(sample["text"])  # 对文本进行清洗
            sample["text"], mapping = clean_text_and_generate_mapping(sample["text"])  # 对文本进行清洗
            if data_type == "test":
                continue
            for rel in sample["relation_list"]:
                # rel["subject"] = clean_text(rel["subject"])
                # rel["object"] = clean_text(rel["object"])
                rel["subject"], _ = clean_text_and_generate_mapping(rel["subject"])
                rel["object"], _ = clean_text_and_generate_mapping(rel["object"])
                if rel.get("subj_char_span"):   # JYZ add
                    rel['subj_char_span'] = text_clean_pos_mapping(rel['subj_char_span'], mapping)
                if rel.get("obj_char_span"):
                    rel['obj_char_span'] = text_clean_pos_mapping(rel['obj_char_span'], mapping)
            for ent in sample['entity_list']:
                ent['text'], _ = clean_text_and_generate_mapping(ent['text'])
                if ent.get('char_span'):
                    ent['char_span'] = text_clean_pos_mapping(ent['char_span'], mapping)
            if sample.get('relation_list_original'):  # for JYZ's dataset
                for triple in sample['relation_list_original']:
                    triple[0][0], _ = clean_text_and_generate_mapping(triple[0][0])  # subject
                    triple[0][2], _ = clean_text_and_generate_mapping(triple[0][2])  # object
                    for i in range(len(triple[1])):    # subject pos
                        triple[1][i] = text_clean_pos_mapping(triple[1][i], mapping)
                    for i in range(len(triple[2])):  # object pos
                        triple[2][i] = text_clean_pos_mapping(triple[2][i], mapping)

        return ori_data

    def clean_data_w_span(self, ori_data):
        '''
        stripe whitespaces and change spans
        add a stake to bad samples(char span error) and remove them from the clean data
        '''
        bad_samples, clean_data = [], []

        def strip_white(entity, entity_char_span):
            p = 0
            while entity[p] == " ":
                entity_char_span[0] += 1
                p += 1

            p = len(entity) - 1
            while entity[p] == " ":
                entity_char_span[1] -= 1
                p -= 1
            return entity.strip(), entity_char_span

        for sample in tqdm(ori_data, desc="clean data w char spans"):
            text = sample["text"]

            bad = False
            for rel in sample["relation_list"]:
                # rm whitespaces
                rel["subject"], rel["subj_char_span"] = strip_white(rel["subject"], rel["subj_char_span"])
                rel["object"], rel["obj_char_span"] = strip_white(rel["object"], rel["obj_char_span"])

                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                if rel["subject"] not in text or rel["subject"] != text[subj_char_span[0]:subj_char_span[1]] or \
                        rel["object"] not in text or rel["object"] != text[obj_char_span[0]:obj_char_span[1]]:
                    rel["stake"] = 0
                    bad = True

            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_rel_list = [rel for rel in sample["relation_list"] if "stake" not in rel]
            if len(new_rel_list) > 0:
                sample["relation_list"] = new_rel_list
                clean_data.append(sample)
        return clean_data, bad_samples

    def _get_char2tok_span(self, text):
        '''
        map character index to token level span
        '''
        tok2char_span = self._get_tok2char_span_map(text)
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
        return char2tok_span

    def _get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        '''
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                if not ignore_subword_match and re.match("\d+", target_ent):  # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            #             if len(spans) == 0:
            #                 set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans

    def add_char_span(self, dataset, ignore_subword_match=True):
        miss_sample_list = []
        for sample in tqdm(dataset, desc="adding char level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities, ignore_subword_match=ignore_subword_match)
            ##### 获取实体在句中的跨度位置

            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_char_spans = ent2char_spans[rel["subject"]]
                obj_char_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_char_spans:
                    for obj_sp in obj_char_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                        })

            if len(sample["relation_list"]) > len(new_relation_list):
                miss_sample_list.append(sample)
            sample["relation_list"] = new_relation_list

            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                sample["entity_list"] = new_ent_list
        return dataset, miss_sample_list

    def add_tok_span(self, dataset):
        """
        dataset must has char level span
        """

        def char_span2tok_span(char_span, char2tok_span):
            # char2tok_span: 列表，每个元素表示每个句中字符对应的token下标。
            #   每个元素一般取值为[a,a+1]，
            #   如果连续多个元素位于一个token中，则会出现`[a,a+1],[a,a+1],...`，
            #   如果是例如空格等字符，不会出现在token中，则取值[-1,-1]

            # print(f"char_span={char_span}")
            # print(f"char2tok_span={char2tok_span}")
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            # print(f"tok_span_list={tok_span_list}")
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for sample in tqdm(dataset, desc="adding token level spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                # print(f"sample['text']={sample['text']}")
                # print(f"rel={rel}")
                # print("subj")
                rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
                # print("obj")
                rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                # print('ent')
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = char_span2tok_span(event["trigger_char_span"], char2tok_span)
                    for arg in event["argument_list"]:
                        arg["tok_span"] = char_span2tok_span(arg["char_span"], char2tok_span)
        return dataset