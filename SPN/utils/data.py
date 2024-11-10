from utils.alphabet import Alphabet
import os, pickle, copy, sys, copy, time
from utils.functions import data_process, data_process_2407
try:
    from transformers import BertTokenizer
except:
    from pytorch_transformers import BertTokenizer


class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.weight = {}

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args, data_process, tokenizer):

        if "train_file" in args:   # True absolutly
            self.train_loader = data_process(args.train_file, self.relational_alphabet, tokenizer, args)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
        if "valid_file" in args:
            self.valid_loader = data_process(args.valid_file, self.relational_alphabet, tokenizer, args)
        if "test_file" in args:
            self.test_loader = data_process(args.test_file, self.relational_alphabet, tokenizer, args)
            print(f"\n-- data example:\n{self.test_loader[0]}\n")
            time.sleep(10)
        self.relational_alphabet.close()


def build_data(args, tokenizer):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    # print(file)  # ./data/generated_data/WebNLG_Set-Prediction-Networks_data.pickle
    ##### file 是经过 tokenizer 预处理后的打包数据

    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(args)   # pickle 文件已存在，无需重新生成

    else:
        # 生成 pickle 文件
        data = Data()        # 初始化对象 <data>
        data.generate_instance(args, data_process_2407, tokenizer=tokenizer)  # process
        save_data_setting(data, args)     # 将预处理后的数据 <data> 保存

    print("-- relations:")
    print(data.relational_alphabet.instance2index)
    print(data.relational_alphabet.instances)
    print("")
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data

