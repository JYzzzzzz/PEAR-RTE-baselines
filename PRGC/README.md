# PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction
This repository contains the source code and dataset for the paper: **PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction.** Hengyi Zheng, Rui Wen, Xi Chen et al. ACL 2021.

## Overview

![image-20210622212609011](https://raw.githubusercontent.com/hy-struggle/img/master/markdown/20210622212609.png)

## Requirements

The main requirements are:

  - python==3.7.9
  - pytorch==1.6.0
  - transformers==3.2.0
  - tqdm

## Datasets

- [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT) and [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)(following [CasRel](https://github.com/weizhepei/CasRel))
- [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)(following [CopyRE](https://github.com/xiangrongzeng/copy_re))
- [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)(following [ETL-span](https://github.com/yubowen-ph/JointER))

Or you can just download our preprocessed [datasets](https://drive.google.com/file/d/1hpUedGxzpg6lyNemClfMCeTXeaBBQ1u7/view?usp=sharing).

## Usage

**1. Get pre-trained BERT model for PyTorch**

Download [BERT-Base-Cased](https://huggingface.co/bert-base-cased/tree/main) which contains `pytroch_model.bin`, `vocab.txt` and `config.json`. Put these under `./pretrain_models`.

**2. Build Data**

Put our preprocessed datasets under `./data`.

**3. Train**

Just run the script in `./script` by `sh train.sh`.

For example, to train the model for NYT* dataset, update the `train.sh` as:

```
python ../train.py \
--ex_index=1 \
--epoch_num=100 \
--device_id=0 \
--corpus_type=NYT-star \
--ensure_corres \
--ensure_rel
```

**4. Evaluate**

Just run the script in `./script` by `sh evaluate.sh`.

For example, to train the model for NYT* dataset, update the `evaluate.sh` as:

```
python ../evaluate.py \
--ex_index=1 \
--device_id=0 \
--mode=test \
--corpus_type=NYT-star \
--ensure_corres \
--ensure_rel \
--corres_threshold=0.5 \
--rel_threshold=0.1
```

# Change for CMIM23-NOM1-RA

- the place about tokenizing, checkpoint output, and some default parameters.
- any place with the note `jyz chg`.
- CMIM23-NOM1-RA is placed in `./data/CMIM23-NOM1-RA`


## How to use after change

1. Check parameters in `train.sh`, `train.py`, and `utils.py`, especially the parameters related to *pre-trained model path* and *output path*.

2. Prepare the pre-trained model and place it in *pre-trained model path* (`pretrain_models/chinese-bert-wwm-ext` is default). You can choose "[chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)" like PEAR or use another one.

3. Train
```shell
cd script
bash train.sh
```

4. Check output in *output path* (`experiments` is default).


## Source code URL
(https://github.com/hy-struggle/PRGC)

