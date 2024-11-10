#!/bin/bash

pypath=$(dirname $(dirname $(readlink -f $0)))
echo "pypath = ${pypath}"

#num_negs_list=(3 )

# 241012 = 240901_lrratio0.2_dropout0.2
output_dir_list=(
"experiments/241012"
)

# Multiple experiments conducted in series
echo "shell: output_dir_list=${output_dir_list[*]}"
for i in "${!output_dir_list[@]}"
do
#  echo "shell: --num_negs=${num_negs_list[$((i))]}"
  echo "shell: --output_dir=${output_dir_list[$((i))]}"

  python $pypath/train.py \
    --batch_size=8 \
    --lr_ratio=0.2 \
    --dropout=0.2 \
    --output_dir="${output_dir_list[$((i))]}" \
    --epoch_num=100 \
    --corpus_type=NYT \
    --ensure_corres \
    --ensure_rel \
    --ex_index=1 \
    --device_id=1 \
#    --num_negs=${num_negs_list[$((i))]} \

  # jyz chg
  cd ..
  python get_score.py \
      --MODEL_NAME="PRGC" \
      --PRETRAIN_MODEL_DIR="pretrain_models/chinese-bert-wwm-ext" \
      --DATASET_DIR="data/CMIM23-NOM1-RA" \
      --LABEL_FILENAME_dev="val_triples.json" --LABEL_FILENAME_test="test_triples.json"\
      --OUTPUT_DIR="${output_dir_list[$((i))]}" \
      --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \
  #    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

  python get_score.py \
      --MODEL_NAME="PRGC" \
      --PRETRAIN_MODEL_DIR="pretrain_models/chinese-bert-wwm-ext" \
      --DATASET_DIR="data/CMIM23-NOM1-RA" \
      --LABEL_FILENAME_dev="val_triples.json" --LABEL_FILENAME_test="test_triples.json"\
      --OUTPUT_DIR="${output_dir_list[$((i))]}" \
      --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \
      --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

#  cd script


done



