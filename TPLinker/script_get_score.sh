#!/bin/bash

# 如果是在超算上跑，需要先 source xxx
if [[ $(pwd) =~ "u2021110308" ]]   # 路径包含某字符串
then
  echo "bupt super computer"
  # shellcheck disable=SC1090
  source ~/.bashrc  ### 初始化环境变量
  source  /opt/app/anaconda3/bin/activate python3_10
fi

output_dir="tplinker/default_log_dir/240914_epo70_lr3e-5"

python3 get_score.py \
    --MODEL_NAME="tplinker" \
    --PRETRAIN_MODEL_DIR="models/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --PREDICT_FILENAME_dev="prediction_valid.json" --PREDICT_FILENAME_test="prediction_test.json" \
#    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

python3 get_score.py \
    --MODEL_NAME="tplinker" \
    --PRETRAIN_MODEL_DIR="models/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --PREDICT_FILENAME_dev="prediction_valid.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

