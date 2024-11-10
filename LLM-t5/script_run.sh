#!/bin/bash

config_file="config/Randeng784M.yaml"

output_dir=$(awk -F ': ' '/^output_dir:/{print $2}' ${config_file})   # read from config_file
output_dir="${output_dir## }"
echo "config_file=${config_file}"
echo "output_dir=${output_dir}"

# 先训练所有checkpoint，然后再统一推理得到预测案例输出
python finetune240517.py --config_file ${config_file} --train_or_infer train_infer

# 将预测案例进行初步整合
python answer_integrate.py --config_file ${config_file}

# get score 打分。注意要将dataset路径写为原数据集路径，pretrain model 采用 chinese-bert-wwm-ext
bert_path="models/chinese-bert-wwm-ext"
dataset_path_for_score="data/CMIM23-NOM1-RA"

python get_score.py \
  --MODEL_NAME="t5" \
  --PRETRAIN_MODEL_DIR=${bert_path} \
  --DATASET_DIR="${dataset_path_for_score}" \
  --OUTPUT_DIR="${output_dir}" \
  --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
  --PREDICT_FILENAME_dev="dataset_prediction_dev_integ.json" \
  --PREDICT_FILENAME_test="dataset_prediction_test_integ.json" \
  --llm_output_group="0,1,2,3,4,5,6,7,8,9" \
  # --DATASET_DIR="../ner_code_2311/ner_code_231117/dataset/CMIM23-NOM1-RA" \

python get_score.py \
  --MODEL_NAME="t5" \
  --PRETRAIN_MODEL_DIR=${bert_path} \
  --DATASET_DIR="${dataset_path_for_score}" \
  --OUTPUT_DIR="${output_dir}" \
  --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
  --PREDICT_FILENAME_dev="dataset_prediction_dev_integ.json" \
  --PREDICT_FILENAME_test="dataset_prediction_test_integ.json" \
  --llm_output_group="0,1,2,3,4,5,6,7,8,9" \
  --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
  # --DATASET_DIR="../ner_code_2311/ner_code_231117/dataset/CMIM23-NOM1-RA" \



