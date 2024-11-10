#!/bin/bash

max_span_length_list=(50 )
num_generated_triples_list=(20 )
na_rel_coef_list=(0.25 )
encoder_lr_list=(3e-5 )
decoder_lr_list=(6e-5 )
output_dir_list=(
"outputs/SPN_240914_SpanLen50_TriNum20_EnLR3e-5"
)

for i in "${!output_dir_list[@]}"
do
  echo "--output_dir=${output_dir_list[$((i))]}"

  python -m main \
      --max_span_length ${max_span_length_list[$((i))]} \
      --num_generated_triples ${num_generated_triples_list[$((i))]} \
      --na_rel_coef ${na_rel_coef_list[$((i))]} \
      --encoder_lr ${encoder_lr_list[$((i))]} \
      --decoder_lr ${decoder_lr_list[$((i))]} \
      --output_dir ${output_dir_list[$((i))]} \
      --max_epoch 80 \
      --visible_gpu 0 \

  python get_score.py \
      --MODEL_NAME="SPN" \
      --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
      --DATASET_DIR="data/CMIM23-NOM1-RA" \
      --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
      --OUTPUT_DIR=${output_dir_list[$((i))]} \
      --PREDICT_FILENAME_dev="dataset_prediction_dev.json" \
      --PREDICT_FILENAME_test="dataset_prediction_test.json" \
  #    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

  python get_score.py \
      --MODEL_NAME="SPN" \
      --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
      --DATASET_DIR="data/CMIM23-NOM1-RA" \
      --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
      --OUTPUT_DIR=${output_dir_list[$((i))]} \
      --PREDICT_FILENAME_dev="dataset_prediction_dev.json" \
      --PREDICT_FILENAME_test="dataset_prediction_test.json" \
      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

## delete model file
#  echo "delete model in ${output_dir_list[$((i))]}"
#  find ${output_dir_list[$((i))]} -type f -name 'model_params.pt' | xargs rm -f

done
