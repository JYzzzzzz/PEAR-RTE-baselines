import json
import os
from tqdm import tqdm
import re
from transformers import BertTokenizerFast
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
from utils import Preprocessor
# from ..tplinker.common
import yaml
import logging
from pprint import pprint
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
# config = yaml.load(open("preprocess/build_data_config.yaml", "r"), Loader = yaml.FullLoader)
config = yaml.load(open("build_data_config.yaml", "r"), Loader = yaml.FullLoader)
exp_name = config["exp_name"]
data_in_dir = os.path.join(config["data_in_dir"], exp_name)
data_out_dir = os.path.join(config["data_out_dir"], exp_name)
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)
file_name2data = {}    # 存放数据集数据
for path, folds, files in os.walk(data_in_dir):
    for file_name in files:
        file_path = os.path.join(path, file_name)
        print(file_name)
        res = re.match("(.*?)\.json", file_name)
        if res is not None:
            file_name = res.group(1)
            print(" ", file_name)
            file_name2data[file_name] = json.load(open(file_path, "r", encoding = "utf-8"))
print("data")
print(f"list(file_name2data.keys()) = \n  {list(file_name2data.keys())}")
print(f"len(file_name2data['train_data']) = \n  {len(file_name2data['train_data'])}")
print(f"example of file_name2data['train_data'] = \n  {file_name2data['train_data'][0]}")
time.sleep(5)

# @specific
if config["encoder"] == "BERT":
    print("loading bert tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]
elif config["encoder"] == "BiLSTM":
    tokenize = lambda text: text.split(" ")
    def get_tok2char_span_map(text):
        tokens = tokenize(text)
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1 # +1: whitespace
        return tok2char_span
preprocessor = Preprocessor(tokenize_func = tokenize, 
                            get_tok2char_span_map_func = get_tok2char_span_map)
ori_format = config["ori_data_format"]
if ori_format != "tplinker": # if tplinker, skip transforming
    for file_name, data in file_name2data.items():
        if "train" in file_name:
            data_type = "train"
        if "valid" in file_name:
            data_type = "valid"
        if "test" in file_name:
            data_type = "test"
        data = preprocessor.transform_data(data, ori_format = ori_format, dataset_type = data_type, add_id = True)
        file_name2data[file_name] = data


def check_tok_span(data):
    def extr_ent(text, tok_span, tok2char_span):
        char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
        char_span = (char_span_list[0][0], char_span_list[-1][1])
        decoded_ent = text[char_span[0]:char_span[1]]
        return decoded_ent

    span_error_memory = set()
    for sample in tqdm(data, desc = "check tok spans"):
        text = sample["text"]
        tok2char_span = get_tok2char_span_map(text)
        for ent in sample["entity_list"]:
            tok_span = ent["tok_span"]
            if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                span_error_memory.add("extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span), ent["text"]))
                
        for rel in sample["relation_list"]:
            subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
            if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span), rel["subject"]))
            if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span), rel["object"]))
    return span_error_memory


# clean, add char span, tok span
# collect relations
# check tok spans
rel_set = set()
ent_set = set()
error_statistics = {}
for file_name, data in file_name2data.items():
    assert len(data) > 0
    if "relation_list" in data[0]: # train or valid data
        # rm redundant whitespaces
        # separate by whitespaces, clean data
        data = preprocessor.clean_data_wo_span(data, separate = config["separate_char_by_white"])
        error_statistics[file_name] = {}
#         if file_name != "train_data":
#             set_trace()
        # ------------------------------ add char span
        if config["add_char_span"]:
            data, miss_sample_list = preprocessor.add_char_span(data, config["ignore_subword"])
            error_statistics[file_name]["miss_samples"] = len(miss_sample_list)
            """
            data[0] = {
                'id': 0, 
                'text': '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名
                        小说电视剧《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说
                        《裸婚时代》是月影兰析创作的一部情感小说《琅琊榜》是根据海宴同名网络小说改编电视剧
                        《宫锁心玉》，又名《宫》《雪豹》，该剧改编自网络小说《特战先驱》《我是特种兵》由
                        红遍网络的小说《最后一颗子弹留给我》改编电视剧《来不及说我爱你》改编自匪我思存同
                        名小说《来不及说我爱你》', 
                'relation_list': [
                    {'subject': '何以笙箫默', 'object': '顾漫', 'subj_char_span': (44, 49), 'obj_char_span': (53, 55), 'predicate': '作者'}, 
                    {'subject': '我是特种兵', 'object': '最后一颗子弹留给我', 'subj_char_span': (152, 157), 'obj_char_span': (167, 176), 'predicate': '改编自'}, 
                    {'subject': '步步惊心', 'object': '桐华', 'subj_char_span': (1, 5), 'obj_char_span': (13, 15), 'predicate': '作者'}, 
                    ...
                ], 
                'entity_list': [
                    {'text': '顾漫', 'type': '人物', 'char_span': (53, 55)}, 
                    {'text': '何以笙箫默', 'type': '图书作品', 'char_span': (44, 49)}, 
                    {'text': '最后一颗子弹留给我', 'type': '作品', 'char_span': (167, 176)}, 
                    ...
                ]
            }
            """

        #         # clean
#         data, bad_samples_w_char_span_error = preprocessor.clean_data_w_span(data)
#         error_statistics[file_name]["char_span_error"] = len(bad_samples_w_char_span_error)

        # ------------------------------ statistic relation types and entity types
        for sample in tqdm(data, desc = "building relation type set and entity type set"):
            # if "entity_list" not in sample, generate entity list with default type
            if "entity_list" not in sample:
                ent_list = []
                for rel in sample["relation_list"]:
                    ent_list.append({
                        "text": rel["subject"],
                        "type": "DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })    # default type
                    ent_list.append({
                        "text": rel["object"],
                        "type": "DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                sample["entity_list"] = ent_list

            # 统计实体类型
            for ent in sample["entity_list"]:
                ent_set.add(ent["type"])
                for ent in sample["entity_list"]:
                    ent_set.add(ent["type"])

            # 统计关系类型
            for rel in sample["relation_list"]:
                rel_set.add(rel["predicate"])
               
        # ------------------------------ add tok span, adding token level spans
        data = preprocessor.add_tok_span(data)
        # data = preprocessor.add_tok_span(data[:100])
        """
        总之就是添加了 tok_span
        data[0] = {
            'id': 0, 
            'text': '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名小说电视剧...', 
            'relation_list': [
                {'subject': '何以笙箫默', 'object': '顾漫', 'subj_char_span': (44, 49), 
                            'obj_char_span': (53, 55), 'predicate': '作者', 
                            'subj_tok_span': [44, 49], 'obj_tok_span': [53, 55]}, 
                {'subject': '步步惊心', 'object': '桐华', 'subj_char_span': (1, 5), 
                            'obj_char_span': (13, 15), 'predicate': '作者', 
                            'subj_tok_span': [1, 5], 'obj_tok_span': [13, 15]}, 
                ...
            ], 
            'entity_list': [
                {'text': '顾漫', 'type': '人物', 'char_span': (53, 55), 'tok_span': [53, 55]}, 
                {'text': '何以笙箫默', 'type': '图书作品', 'char_span': (44, 49), 'tok_span': [44, 49]}, 
                ...
            ]
        }
        """
        print(data[0])
        # print(xxxxx)

        # check tok span
        if config["check_tok_span"]:
            span_error_memory = check_tok_span(data)
            if len(span_error_memory) > 0:
                print("token span error:")
                print(span_error_memory)
            error_statistics[file_name]["tok_span_error"] = len(span_error_memory)
            
        file_name2data[file_name] = data

pprint(error_statistics)
rel_set = sorted(rel_set)
rel2id = {rel:ind for ind, rel in enumerate(rel_set)}

ent_set = sorted(ent_set)
ent2id = {ent:ind for ind, ent in enumerate(ent_set)}

data_statistics = {
    "relation_type_num": len(rel2id),
    "entity_type_num": len(ent2id),
}

for file_name, data in file_name2data.items():
    data_path = os.path.join(data_out_dir, "{}.json".format(file_name))
    json.dump(data, open(data_path, "w", encoding = "utf-8"), ensure_ascii = False, indent=4)
    logging.info("{} is output to {}".format(file_name, data_path))
    data_statistics[file_name] = len(data)

rel2id_path = os.path.join(data_out_dir, "rel2id.json")
json.dump(rel2id, open(rel2id_path, "w", encoding = "utf-8"), ensure_ascii = False, indent=4)
logging.info("rel2id is output to {}".format(rel2id_path))

ent2id_path = os.path.join(data_out_dir, "ent2id.json")
json.dump(ent2id, open(ent2id_path, "w", encoding = "utf-8"), ensure_ascii = False, indent=4)
logging.info("ent2id is output to {}".format(ent2id_path))

data_statistics_path = os.path.join(data_out_dir, "data_statistics.txt")
json.dump(data_statistics, open(data_statistics_path, "w", encoding = "utf-8"), ensure_ascii = False, indent = 4)
logging.info("data_statistics is output to {}".format(data_statistics_path)) 
pprint(data_statistics)