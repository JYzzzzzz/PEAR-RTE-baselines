import string
import random
import time

common = {
    # "exp_name": "baidu_relation",    # !!!!! part of data dir
    "exp_name": "CMIM23-NOM1-RA",
    "rel2id": "rel2id.json",
    # "device_num": 2,3,
#     "encoder": "BiLSTM",
    "encoder": "BERT", 
    "hyper_parameters": {
        "shaking_type": "cat", # cat, cat_plus, cln, cln_plus; Experiments show that cat/cat_plus work better with BiLSTM, while cln/cln_plus work better with BERT. The results in the paper are produced by "cat". So, if you want to reproduce the results, "cat" is enough, no matter for BERT or BiLSTM.
        "inner_enc_type": "lstm", # valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between each token pairs. If you only want to reproduce the results, just leave it alone.
        "dist_emb_size": -1, # -1: do not use distance embedding; other number: need to be larger than the max_seq_len of the inputs. set -1 if you only want to reproduce the results in the paper.
        "ent_add_dist": False, # set true if you want add distance embeddings for each token pairs. (for entity decoder)
        "rel_add_dist": False, # the same as above (for relation decoder)
        # "match_pattern": "only_head_text", # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span
        "match_pattern": "whole_text",
    },
}
common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

# run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
run_id = f"run_{time.strftime('%Y%m%d_%H%M')}"

train_config = {
    "train_data": "train_data.json",    # 完整路径为 data_home + exp_name + train_data
    "valid_data": "valid_data.json",
    "test_data": "test_data.json",
    "train_test_data": "v_test.json",
    "rel2id": "rel2id.json",

    # # if logger is set as wandb, comment the following four lines
    # "logger": "wandb",
    # if logger is set as default, uncomment the following four lines
    "logger": "default",
    "run_id": run_id,
    "log_path": f"./default_log_dir/log_{time.strftime('%Y%m%d_%H%M')}.log",

    "path_to_save_model": "./default_log_dir/240914_epo70_lr3e-5",    # !!!!! 输出文件夹

    # only save the model state dict if F1 score surpasses <f1_2_save>
    "f1_2_save": 0.2,
    # whether train_config from scratch
    "fr_scratch": True,
    # write down notes here if you want, it will be logged 
    "note": "start from scratch",
    # if not fr scratch, set a model_state_dict
    "model_state_dict_path": "",
    "gpu_id": "0",                   # !!!!!
    "hyper_parameters": {
        # "batch_size": 25,    # !!!!! train batch size
        "batch_size": 8,
        "epochs": 70,          # 50 origin !!!!!
        "seed": 2333,
        "log_interval": 100,
        # "max_seq_len": 100,    # !!!!! 输入句子的最长token数
        "max_seq_len": 200,
        "sliding_len": 50,    # 好像是用于当句子超过最长长度时，除了截掉最后一段，还会将窗口向后滑动以获得新的句子。
        "loss_weight_recover_steps": 6000,   # to speed up the training process, the loss of EH-to-ET sequence is set higher than other sequences at the beginning, but it will recover in <loss_weight_recover_steps> steps.
        "scheduler": "CAWR", # Step
    },
}

eval_config = {
    "model_state_dict_dir": "../wandb", # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    # "run_ids": ["10suiyrf", ],
    "run_ids": ["files"],
    "last_k_model": 1,
    # "test_data": "*test*.json", # "*test*.json"
    "test_data": "valid_data.json",
    
    # where to save results
    "save_res": False,
    "save_res_dir": "../results",
    
    # score: set true only if test set is annotated with ground truth
    "score": True,
    
    "hyper_parameters": {
        "batch_size": 8,
        "force_split": False,
        "max_test_seq_len": 512,
        "sliding_len": 50,
    },
}

# bert_config = {
#     "data_home": "/home/yuanchaoyi/TPlinker-joint-extraction/data4bert",
#     "bert_path": "/home/yuanchaoyi/BeiKe/QA_match/roberta_base",
#     "hyper_parameters": {
#         "lr": 5e-5,
#     },
# }
bert_config = {
    "data_home": "../data4bert",
    "bert_path": "../models/chinese-bert-wwm-ext",   # !!!!!
        # 修改预训练模型时，需要改BuildData.py中的参数，并重新生成data4bert下的训练文件
    "hyper_parameters": {
        "lr": 3e-5,    # 5e-5 origin !!!!!
    },
}

bilstm_config = {
    "data_home": "../data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "../../pretrained_emb/glove_300_nyt.emb",
    "hyper_parameters": {
         "lr": 1e-3,
         "enc_hidden_size": 300,
         "dec_hidden_size": 600,
         "emb_dropout": 0.1,
         "rnn_dropout": 0.1,
         "word_embedding_dim": 300,
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------
if common["encoder"] == "BERT":
    hyper_params = {**common["hyper_parameters"], **bert_config["hyper_parameters"]}
    common = {**common, **bert_config}
    common["hyper_parameters"] = hyper_params
elif common["encoder"] == "BiLSTM":
    hyper_params = {**common["hyper_parameters"], **bilstm_config["hyper_parameters"]}
    common = {**common, **bilstm_config}
    common["hyper_parameters"] = hyper_params
    
hyper_params = {**common["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}
    
hyper_params = {**common["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common}
eval_config["hyper_parameters"] = hyper_params
