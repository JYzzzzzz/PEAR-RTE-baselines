#!/bin/bash

# need GPU about 40GB

sample_num=3337    # const. sample number in training dataset of CMIM23-NOM1-RA
checkpoint_num=50    # const
batch_size=4     # varb
epoch=100        # varb
#save_steps=$((sample_num * epoch / (batch_size * checkpoint_num)))
save_steps=$((sample_num  / batch_size ))
echo "--save_steps=${save_steps}"



output_dir="outputs/240914"

python3 train.py \
  --lr=1e-5 \
  --batch_size=${batch_size} \
  --max_epoch=${epoch} \
  --eval_step=${save_steps} \
  --output_dir=${output_dir} \
  --gpu_id="0"

python3 get_score.py \
    --MODEL_NAME="OneRel" \
    --PRETRAIN_MODEL_DIR="pre_trained_bert/chinese-bert-wwm-ext" \
    --DATASET_DIR="data/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="dev_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-step" \
    --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \

python3 get_score.py \
    --MODEL_NAME="OneRel" \
    --PRETRAIN_MODEL_DIR="pre_trained_bert/chinese-bert-wwm-ext" \
    --DATASET_DIR="data/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="dev_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-step" \
    --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

echo "delete model in ${output_dir}"
find ${output_dir} -type f -name 'MODEL_OneRel*' | xargs rm -f


