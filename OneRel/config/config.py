class Config(object):
    def __init__(self, args):
        self.args = args

        self.pretrain_model_path = "pre_trained_bert/chinese-bert-wwm-ext"
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num
        self.bert_max_len = args.bert_max_len
        self.bert_dim = 768
        self.tag_size = 4
        self.dropout_prob = args.dropout_prob
        self.entity_pair_dropout = args.entity_pair_dropout

        # dataset
        self.dataset = args.dataset

        # path and name
        self.data_path = './data/' + self.dataset
        # self.checkpoint_dir = './checkpoint/' + self.dataset
        self.checkpoint_dir = args.output_dir    # jyz chg. 现 output dir
        self.log_dir = self.checkpoint_dir    # './log/' + self.dataset
        self.result_dir = './result/' + self.dataset    # jyz chg. 原来的 output dir，现弃用
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
        # 数据集完整路径为：data_path + train_prefix + '.json'

        self.model_save_name = 'MODEL_' + args.model_name + '_DATASET_' + self.dataset[-12:] + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len) + "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset[-12:] + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len) + "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset[-12:] + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len)+ "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout) + ".json"

        # log setting
        self.log_step = args.log_step
        self.test_epoch = args.test_epoch
        self.eval_step = args.eval_step   # save_step too

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix
