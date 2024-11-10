"""

version before 240929:
    -- 原本是将yaml文件读取到的参数都传入函数，现改为传入需要的参数
"""


import argparse
import yaml
import os
import time
import json


RELATION_CH2EN = {
    "功能": "Function", "手段采用": "Approach", "前提是": "Premise", "造成": "Cause", "影响": "Influence",
    "分类": "Category", "组成部分": "Component", "属性有": "Attribute", "含有": "Contain",
    "定义": "Definition", "别名": "Alias", "实例为": "Instance", "特点": "Characteristic",
}
RELATION_EN2CH = {}
for ch, en in RELATION_CH2EN.items():
    RELATION_EN2CH[en] = ch
    RELATION_EN2CH[ch] = ch

# 数据集文本中用于分隔三元组的字符串。由于数据集先后做过调整，因此符号不统一
SPECIAL_STR = {
    'begin': '(b)',   # 用于 “rela: subj, obj, subj, obj.” 的格式
    'end': '(e)',
    'subj': '(s)',
    'obj': '(o)',
    'rela': '(r)'
}


def span_find(input_str, symbol_l_str, symbol_r_str, start=0, no_symbol_l_str_in_sub_str=True):
    """
    find the next sub-string between "span_l_str" and "span_r_str" in "input_str"
    version: 240921

    :param input_str:
    :param symbol_l_str: left boundary symbol of span
    :param symbol_r_str: right boundary symbol of span
    :param start: starting position for search
    :param no_symbol_l_str_in_sub_str: 是否要求子串中不含 `span_l_str`
        如果设为False。则 input_str="abcdab--yz", span_l_str="ab", span_r_str="yz" 时，
            sub_str="cdab--". 设为True时，sub_str="--"
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

    symbol_l_pos = input_str.find(symbol_l_str, start)  # find the position of left boundary symbol of span
    if symbol_l_pos < 0:
        return "", -1, -1
    sub_pos_l = symbol_l_pos + len(symbol_l_str)

    symbol_r_pos = input_str.find(symbol_r_str, sub_pos_l)  # find the position of right boundary symbol of span
    if symbol_r_pos < 0:
        return "", -1, -1
    sub_pos_r = symbol_r_pos

    sub_str = input_str[sub_pos_l:sub_pos_r]

    symbol_l_pos2 = sub_str.find(symbol_l_str)
    while no_symbol_l_str_in_sub_str is True and symbol_l_pos2 > -1:
        # 截掉前缀，确保子串中没有span_l_str
        sub_pos_l += symbol_l_pos2 + len(symbol_l_str)
        sub_str = sub_str[symbol_l_pos2 + len(symbol_l_str):]
        symbol_l_pos2 = sub_str.find(symbol_l_str)

    return sub_str, sub_pos_l, sub_pos_r


def get_triples_in_ans(ans, sample_ans_mode):
    """

    :param ans:
    :param sample_ans_mode: 区分不同的回答格式，用于确定整合的模式。
    :return:  ["subj[sep]rela[sep]obj", ...]
    """
    triples_set = set()

    if 'rela_order__sample_expand__oneshot/' in sample_ans_mode:
        rela_pos = 0
        while 1:
            # 提取具有相同关系的主客体对的字符串
            rela_triples_str, _, rela_pos = span_find(ans, "b) ", " e)", rela_pos)
            rela_end = rela_triples_str.find(":")
            if rela_end == -1:  # rela字符串不完整，或者 rela_triples_str==""
                break

            # 获得rela的最终表示
            rela = rela_triples_str[:rela_end].strip()
            if rela not in list(RELATION_EN2CH.keys()):
                print(f"    rela not in set: \nans = {ans}\ntriple_str = {rela_triples_str}")
                continue
            rela = RELATION_EN2CH[rela]

            # -------------------- find triples
            triples_str = rela_triples_str[rela_end:]
            # ^^^ example: ": s) 传送测量报告 o) SACCH信道"
            triple_str_list = triples_str.split(" s) ")
            # ^^^ example: [":", "传送测量报告 o) SACCH信道"]
            for triple_str in triple_str_list:
                if triple_str.count(" o) ") == 0:
                    continue
                subj_obj_list = triple_str.split(" o) ")
                subj = subj_obj_list[0].strip()
                obj = subj_obj_list[1].strip()
                if subj == "" or obj == "":
                    continue
                triple_represent = f"{subj}[sep]{rela}[sep]{obj}"
                triples_set.add(triple_represent)

    elif 'triple_order' in sample_ans_mode:
        # example: "s) SACCH信道 r) 含有 o) 测量报告 e) s) 传送测量报告 r) 手段采用 o) SACCH信道 e) e)"
        triple_pos = 0
        while 1:
            # 提取三元组的字符串
            triple_str, triple_pos, _ = span_find(
                ans, SPECIAL_STR['subj'], SPECIAL_STR['end'], triple_pos)
            if triple_str == "":
                break

            if triple_str.count(SPECIAL_STR['rela']) == 0 or triple_str.count(SPECIAL_STR['obj']) == 0:  # 字符串不完整
                continue

            triple_str = SPECIAL_STR['subj'] + triple_str + SPECIAL_STR['end']
            subj = span_find(triple_str, SPECIAL_STR['subj'], SPECIAL_STR['rela'])[0].strip()
            obj = span_find(triple_str, SPECIAL_STR['obj'], SPECIAL_STR['end'])[0].strip()
            if subj == "" or obj == "":
                continue

            rela = span_find(triple_str, SPECIAL_STR['rela'], SPECIAL_STR['obj'])[0].strip()
            if rela not in list(RELATION_EN2CH.keys()):
                # print(f"    rela not in set: \nans = {ans}\ntriple_str = {triple_str}")
                continue
            rela = RELATION_EN2CH[rela]

            triple_represent = f"{subj}[sep]{rela}[sep]{obj}"
            triples_set.add(triple_represent)
    else:
        assert 0, f"\nsample_ans_mode={sample_ans_mode}"  # 未知的数据集，无法确定整合模式

    return list(triples_set)


def integrate_1_file(file_in, sample_expand_mode, sample_ans_mode,
                     ques_key, pred_key):
    """

    :param file_in:
    :param sample_expand_mode: 数据集中样本扩展的模式。例如：[[13],[13],[13],[13],[13],[13],[13],[13],[13],[13],]
    :param sample_ans_mode: 用来表示数据回答格式。例如："triple_order"
    :return:
    """

    if file_in[-6:] == ".jsonl":
        datas_in = []
        with open(file_in, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)  # 使用 json.loads() 将 JSON 格式的字符串解析为字典
                datas_in.append(entry)  # 将解析后的字典添加到列表中
        # print(f"    example in jsonl: {datas_in[0]}")
    else:    # json
        with open(file_in, 'r', encoding='UTF-8') as f:
            datas_in = json.loads(f.read())
    print(f"    len(datas_in) = {len(datas_in)}")
    # time.sleep(5)

    samples_out = []
    d_i = 0
    while 1:
        # -------------------- 对原数据集中的一个样本进行操作
        text = ""
        triple_pred_dict = {}
        for lg_i, label_group in enumerate(sample_expand_mode):
            # example of label_group: [3, 3, 3, 4]

            # -------------------- 合并一个label_group的ans
            ans = ""
            for _ in label_group:
                ques = datas_in[d_i][ques_key]
                if len(ques) < len(text) or text == "":
                    text = ques   # 找一组中最短的ques作为sent的表示（将就一下）
                ans += datas_in[d_i][pred_key] + SPECIAL_STR['end']   # 额外添加一个 e) 防止结尾没有
                d_i += 1
                if d_i >= len(datas_in):    # 文件中所有回答全部遍历过了
                    break

            # -------------------- find triples
            list_temp = get_triples_in_ans(ans, sample_ans_mode)

            for triple_represent in list_temp:
                if triple_represent not in triple_pred_dict:
                    triple_pred_dict[triple_represent] = []
                if lg_i not in triple_pred_dict[triple_represent]:
                    triple_pred_dict[triple_represent].append(lg_i)

            if d_i >= len(datas_in):  # 文件中所有回答全部遍历过了
                break

        for key in triple_pred_dict.keys():
            triple_pred_dict[key] = str(triple_pred_dict[key])[1:-1]
        sample_out = {
            "text": text,
            "triple_pred_dict": triple_pred_dict,
        }
        samples_out.append(sample_out.copy())

        if d_i >= len(datas_in):    # 文件中所有回答全部遍历过了
            break

    print(f"    samples_out[0] = {samples_out[0]}")
    print(f"    len(samples_out) = {len(samples_out)}")
    file_out = file_in.replace(".jsonl", ".json").replace(".json", "_integ.json")
    print(f"    file_out = {file_out}")
    with open(file_out, "w", encoding="utf-8") as file1:
        json.dump(samples_out, file1, ensure_ascii=False, indent=4)


def integrate_all():

    # ---------------------------------------- parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="../saves/qwen2_liuzhe",)
    parser.add_argument('--prediction_dev', type=str, default="prediction_dev/generated_predictions.jsonl",)
    parser.add_argument('--prediction_test', type=str, default="prediction_test/generated_predictions.jsonl",)
    parser.add_argument('--ques_key', type=str, default="prompt",)
    parser.add_argument('--pred_key', type=str, default="predict",)
    args_from_parser = parser.parse_args()

    # ---------------------------------------- yaml config
    # with open(args_from_parser.config_file, 'r', encoding='utf-8') as file:
    #     args_from_yaml = yaml.safe_load(file)   # a dict
    checkpoint_dir_root = args_from_parser.checkpoint_dir
    sample_expand_mode = [[13] for _ in range(10)]
    sample_ans_mode = "triple_order"

    # get checkpoint dir
    checkpoint_dir_list = []
    folders = list(os.walk(checkpoint_dir_root))[0][1]
    for folder in folders:
        checkpoint_dir_list.append(os.path.join(checkpoint_dir_root, folder))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(5)

    for checkpoint_dir in checkpoint_dir_list:

        file_in = os.path.join(checkpoint_dir, args_from_parser.prediction_dev)
        print(f"\n-- integrating {file_in}")
        if os.path.isfile(file_in) is True:
            integrate_1_file(file_in, sample_expand_mode, sample_ans_mode,
                             args_from_parser.ques_key, args_from_parser.pred_key)

        file_in = os.path.join(checkpoint_dir, args_from_parser.prediction_test)
        print(f"\n-- integrating {file_in}")
        if os.path.isfile(file_in) is True:
            integrate_1_file(file_in, sample_expand_mode, sample_ans_mode,
                             args_from_parser.ques_key, args_from_parser.pred_key)


if __name__ == "__main__":
    integrate_all()
