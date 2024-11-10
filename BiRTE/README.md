# BiRTE
WSDM2022 "A Simple but Effective Bidirectional Extraction Framework for Relational Triple Extraction"

## Requirements
The main requirements are:
- python 3.6
- torch 1.4.0 
- tqdm
- transformers == 2.8.0

## Usage
1. **Train and select the model**

python run.py --dataset=WebNLG  --train=train  --batch_size=6

python run.py --dataset=WebNLG_star  --train=train  --batch_size=6

python run.py --dataset=NYT   --train=train  --batch_size=18

python run.py --dataset=NYT_star   --train=train  --batch_size=18

python run.py --dataset=NYT10   --train=train  --batch_size=18

python run.py --dataset=NYT11   --train=train  --batch_size=18

2. **Evaluate on the test set**

python run.py --dataset=WebNLG --train=test

python run.py --dataset=WebNLG_star --train=test

python run.py --dataset=NYT --train=test

python run.py --dataset=NYT_star --train=test

python run.py --dataset=NYT10 --train=test

python run.py --dataset=NYT11 --train=test


# Change for CMIM23-NOM1-RA (2408)

- 数据集文件添加、预训练模型文件添加、命令行参数路径修改。
- 我的数据集的rel2id文件的格式与代码所需的不同，修改rel2id格式。


- 兼容性修改：transformers库函数调用路径、json打开方式、GPU设备选择。


- main.py - def train() 中，Tokenizer 初始化部分修改。设置将字母小写。
- Tokenizer 改用 Transformer库中的BertTokenizer（原使用本地bert4keras中的），各处都有修改。
- 我的数据集的样本文件的格式与代码所需的不同，临时修改代码:
  - main.py 中新增了一个类 class data_generator_forCMIM2023 专门用于读取我们的数据集。
  - main.py 中新增函数 extract_spoes_forCMIM2023、evaluate_forCMIM2023用于验证过程中操作我们的数据集的数据。
- main.py，train函数中，输出的路径格式调整为自己喜好的，调整较大。
- any positon with note "jyz chg".
- CMIM23-NOM1-RA is placed in `./datasets/CMIM23-NOM1-RA`


## How to use after change

1. Check parameters in `run.py` and `script_run.sh`, especially the parameters related to *pre-trained model path* and *output path*.
2. Prepare the pre-trained model and place it on *pre-trained model path*. You can choose "[chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)" like PEAR or use another one.
3. Train and evaluate: 
```shell
bash script_run.sh
```
4. Check output in *output path*.


## Source code URL
(https://github.com/neukg/BiRTE)
