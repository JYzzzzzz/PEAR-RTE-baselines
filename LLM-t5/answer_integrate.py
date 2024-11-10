
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


def get_triples_in_ans(ans, dataset_path):
    """

    :param ans:
    :param dataset_path: 区分不同的回答格式，用于确定整合的模式。
    :return:  ["subj[sep]rela[sep]obj", ...]
    """
    triples_set = set()

    if 'rela_order__sample_expand__oneshot/' in dataset_path:
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
    elif 'triple_order__sample_expand__oneshot/' in dataset_path:
        # example: "s) SACCH信道 r) 含有 o) 测量报告 e) s) 传送测量报告 r) 手段采用 o) SACCH信道 e) e)"
        triple_pos = 0
        while 1:
            # 提取三元组的字符串
            triple_str, _, triple_pos = span_find(ans, "s) ", " e)", triple_pos)
            if triple_str.count(" o) ") == 0:  # 字符串不完整，或者 rela_triples_str==""
                break

            triple_str = "s) " + triple_str + " e)"
            subj = span_find(triple_str, "s) ", " r) ")[0].strip()
            obj = span_find(triple_str, " o) ", " e)")[0].strip()
            if subj == "" or obj == "":
                continue

            rela = span_find(triple_str, " r) ", " o) ")[0].strip()
            if rela not in list(RELATION_EN2CH.keys()):
                print(f"    rela not in set: \nans = {ans}\ntriple_str = {triple_str}")
                continue
            rela = RELATION_EN2CH[rela]

            triple_represent = f"{subj}[sep]{rela}[sep]{obj}"
            triples_set.add(triple_represent)
    else:
        assert 0, f"\n{dataset_path}"  # 未知的数据集，无法确定整合模式

    return list(triples_set)


def integrate_1_file(file_in, args_from_yaml):
    with open(file_in, 'r', encoding='UTF-8') as f:
        datas_in = json.loads(f.read())
    print(f"    len(datas_in) = {len(datas_in)}")

    samples_out = []
    d_i = 0
    while 1:
        # -------------------- 对原数据集中的一个样本进行操作
        text = ""
        triple_pred_dict = {}
        for lg_i, label_group in enumerate(args_from_yaml['rela_num_distribution']):
            # example of label_group: [3, 3, 3, 4]

            # -------------------- 合并一个label_group的ans
            ans = ""
            for _ in label_group:
                ques = datas_in[d_i]['ques']
                if len(ques) < len(text) or text == "":
                    text = ques   # 找一组中最短的ques作为sent的表示（将就一下）
                ans += datas_in[d_i]['pred_ans'] + " e)"   # 额外添加一个 e) 防止结尾没有
                d_i += 1
                if d_i >= len(datas_in):    # 文件中所有回答全部遍历过了
                    break

            # -------------------- find triples
            list_temp = get_triples_in_ans(ans, args_from_yaml['train_data_path'])

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
    file_out = file_in.replace(".json", "_integ.json")
    print(f"    file_out = {file_out}")
    with open(file_out, "w", encoding="utf-8") as file1:
        json.dump(samples_out, file1, ensure_ascii=False, indent=4)


def integrate_all():

    # ---------------------------------------- parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./config/240921.yaml",
                        )  # 程序运行配置文件，不是预训练模型读取的config
    parser.add_argument('--prediction_dev', type=str, default="dataset_prediction_dev.json",)
    parser.add_argument('--prediction_test', type=str, default="dataset_prediction_test.json",)
    args_from_parser = parser.parse_args()

    # ---------------------------------------- yaml config
    with open(args_from_parser.config_file, 'r', encoding='utf-8') as file:
        args_from_yaml = yaml.safe_load(file)   # a dict

    # get checkpoint dir
    checkpoint_dir_list = []
    folders = list(os.walk(args_from_yaml['output_dir']))[0][1]
    for folder in folders:
        checkpoint_dir_list.append(os.path.join(args_from_yaml['output_dir'], folder))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(5)

    for checkpoint_dir in checkpoint_dir_list:

        file_in = os.path.join(checkpoint_dir, args_from_parser.prediction_dev)
        print(f"\n-- integrating {file_in}")
        if os.path.isfile(file_in) is True:
            integrate_1_file(file_in, args_from_yaml)

        file_in = os.path.join(checkpoint_dir, args_from_parser.prediction_test)
        print(f"\n-- integrating {file_in}")
        if os.path.isfile(file_in) is True:
            integrate_1_file(file_in, args_from_yaml)


if __name__ == "__main__":
    integrate_all()
