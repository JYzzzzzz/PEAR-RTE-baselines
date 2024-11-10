#!/bin/bash

# 必须在 llama-factory 的主目录下运行，不然需要格外设置数据集路径等参数

# config
output_dir_SameAsYaml="./saves/qwen2_240928"
step_list=(1042 2084 3126 4168 5210 6252 7294 8336 9378 10420 11462 12504 13546 14588 15630 16672 17714 18756 19798 20840)
#step_list=(2084 4168 6252 8336 10420 12504 14588 16672 18756 20840)


# train
#CUDA_VISIBLE_DEVICES=0
llamafactory-cli train my_scripts/qwen2_lora_sft.yaml


# predict
for i in "${!step_list[@]}"
do
  checkpoint_path="${output_dir_SameAsYaml}/checkpoint-${step_list[$((i))]}"
  echo "-- checkpoint_path = ${checkpoint_path}"

  # predict a checkpoint on valid(dev) set
  llamafactory-cli train \
      --stage sft --do_predict \
      --model_name_or_path ../qwen/qwen2-7b-instruct \
      --adapter_name_or_path "${checkpoint_path}"  \
      --eval_dataset cmim23_nom1_ra_dev \
      --template qwen --finetuning_type lora \
      --output_dir "${checkpoint_path}/prediction_dev" \
      --overwrite_cache --overwrite_output_dir \
      --cutoff_len 2000 \
      --preprocessing_num_workers 16 --per_device_eval_batch_size 8 \
      --eval_accumulation_steps 1 \
#      --predict_with_generate \
  #    --max_samples 100 \

  # predict a checkpoint on test set
  llamafactory-cli train \
      --stage sft --do_predict \
      --model_name_or_path ../qwen/qwen2-7b-instruct \
      --adapter_name_or_path "${checkpoint_path}"  \
      --eval_dataset cmim23_nom1_ra_test \
      --template qwen --finetuning_type lora \
      --output_dir "${checkpoint_path}/prediction_test" \
      --overwrite_cache --overwrite_output_dir \
      --cutoff_len 2000 \
      --preprocessing_num_workers 16 --per_device_eval_batch_size 8 \
      --eval_accumulation_steps 1 \
#      --predict_with_generate \

done


# integrate
python my_scripts/answer_integrate.py \
  --checkpoint_dir ${output_dir_SameAsYaml} \
  --prediction_dev "prediction_dev/generated_predictions.jsonl" \
  --prediction_test "prediction_test/generated_predictions.jsonl"

# get score
bert_path="my_scripts/bert/chinese-bert-wwm-ext"
dataset_path_for_score="my_scripts/data/CMIM23-NOM1-RA"

python my_scripts/get_score.py \
    --MODEL_NAME="t5" \
    --PRETRAIN_MODEL_DIR="${bert_path}" \
    --DATASET_DIR="${dataset_path_for_score}" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="${output_dir_SameAsYaml}" \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
    --PREDICT_FILENAME_dev="prediction_dev/generated_predictions_integ.json" \
    --PREDICT_FILENAME_test="prediction_test/generated_predictions_integ.json" \
    --llm_output_group="0,1,2,3,4,5,6,7,8,9" \
#    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
#    --PRETRAIN_MODEL_DIR="../ner_code_2311/ner_code_231117/lib/prev_trained_model/chinese-bert-wwm-ext/" \
#    --DATASET_DIR="../ner_code_2311/ner_code_231117/dataset/CMIM23-NOM1-RA/" \

python my_scripts/get_score.py \
    --MODEL_NAME="t5" \
    --PRETRAIN_MODEL_DIR="${bert_path}" \
    --DATASET_DIR="${dataset_path_for_score}" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="${output_dir_SameAsYaml}" \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
    --PREDICT_FILENAME_dev="prediction_dev/generated_predictions_integ.json" \
    --PREDICT_FILENAME_test="prediction_test/generated_predictions_integ.json" \
    --llm_output_group="0,1,2,3,4,5,6,7,8,9" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

