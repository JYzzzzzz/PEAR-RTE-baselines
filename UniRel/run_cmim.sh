export CUDA_VISIBLE_DEVICES=1   # gpu id

sample_num=3337    # const
checkpoint_num=50    # const

# varb
batch_size=8
epoch=50
output_dir="./output/UniRel_240914_batch8_epoch50"

save_steps=$((sample_num * epoch / (batch_size * checkpoint_num)))
#save_steps=$((sample_num  / batch_size ))
echo "--save_steps=${save_steps}"


python3 run.py \
    --per_device_train_batch_size ${batch_size} \
    --num_train_epochs ${epoch} \
    --learning_rate 3e-5 \
    --threshold 0.5 \
    --output_dir ${output_dir} \
    --save_steps ${save_steps} \
    --max_seq_length 200 \
    --dataset_dir "data4bert" --dataset_name "CMIM23-NOM1-RA" \
    \
    --task_name UniRel \
    --per_device_eval_batch_size 16 \
    --logging_dir ./tb_logs \
    --logging_steps 100 \
    --eval_steps 5000000 \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
    --model_dir ./model/chinese-bert-wwm-ext/ \
    --overwrite_output_dir \
    --dataloader_pin_memory \
    --dataloader_num_workers 0 \
    --lr_scheduler_type cosine \
    --seed 2023 \
    --do_test_all_checkpoints\
    --test_data_type unirel_span \
    --do_train


python3 get_score.py \
    --MODEL_NAME="UniRel" \
    --PRETRAIN_MODEL_DIR="model/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
    --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \

python3 get_score.py \
    --MODEL_NAME="UniRel" \
    --PRETRAIN_MODEL_DIR="model/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR=${output_dir} \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-" \
    --PREDICT_FILENAME_dev="prediction_dev.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

#find ${output_dir} -type f -name 'optimizer.pt' | xargs rm -f
#find ${output_dir} -type f -name 'pytorch_model.bin' | xargs rm -f

