#!/bin/bash

output_dir="outputs/240914_SeqLen200_lr3e-5__SegEntDelete"

python3 run.py \
  --max_len 200 \
  --bert_learning_rate=3e-5 \
  --other_learning_rate=15e-5 \
  --train_segment_entity_strategy="delete" \
  --output_dir=${output_dir} \
  --dataset=CMIM23-NOM1-RA  \
  --train=train  \
  --batch_size=8 \
  --cuda_id="1" \


python3 get_score.py \
    --MODEL_NAME='BiRTE' \
    --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
    --DATASET_DIR="datasets/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="dev.json" --LABEL_FILENAME_test="test.json"\
    --OUTPUT_DIR=${output_dir} \
    --PREDICT_FILENAME_dev="dev_pred.json" --PREDICT_FILENAME_test="test_pred.json" \

python3 get_score.py \
    --MODEL_NAME='BiRTE' \
    --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
    --DATASET_DIR="datasets/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="dev.json" --LABEL_FILENAME_test="test.json"\
    --OUTPUT_DIR=${output_dir} \
    --PREDICT_FILENAME_dev="dev_pred.json" --PREDICT_FILENAME_test="test_pred.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \


exit 0



