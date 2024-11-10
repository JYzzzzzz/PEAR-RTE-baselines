# TPLINKER

- TPLINKER注释版本，并适配了中文数据集<br>

- 中文数据来源于百度关系抽取大赛<br>

- 在preprocess路径下build_data_config.yaml中先配置数据源，注意ori_data_format因为用的自己的数据集所以为tplinker<br>

- add_char_span设置为True方便添加char_span<br>

- 在ori_data/baidu_relation/data下dataprocess.py处理百度数据，只处理了一部分数据只是为了方便跑通，看效果<br>

- 在preprocess路径下运行BuildData.py生成数据，结果放置在data4bert/baidu_relation下<br>
- 在tplinker/train_config.yaml配置相应的文件<br>
- 接下来只需运行tplinker下的train.py即可运行。<br>
- 详解说明https://zhuanlan.zhihu.com/p/342300800<br>
- 最近因为工作原因，在搞模型压缩跟联邦学习，等到空闲了把细节补上。感谢~<br>


# Change for CMIM23-NOM1-RA (2406)

- 调整了`tplinker/train.py` 的代码风格（原本由jupter notebook 转化而来，函数排版混乱，存在大量全局变量）
- 在`tplinker/tplinker.py`, `tplinker/train.py` 中添加了验证时返回生成的prediction文本案例的功能，并保存。不影响训练。
- 修改了`tplinker/train.py`中 log, checkpoint, 以及prediction的保存策略。不影响训练。
- 修改了`preprocess/BuildData.py`与`preprocess/utils.py`中有关空白字符调整的程序。适配本课题数据集。
- CMIM23-NOM1-RA is placed in `./data4bert/CMIM23-NOM1-RA`


## How to use after change

1. Check parameters in `tplinker/config.py`, especially the parameters related to *pre-trained model path* and *output path*.

2. Prepare the pre-trained model and place it in *pre-trained model path*. You can choose "[chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)" like PEAR or use another one.

3. Train
```shell
cd tplinker
python3 train.py
```
you can find checkpoints in `tplinker/default_log_dir`(*output path*).

4. Evaluate and get scores. After the training is over, changing the *output path* related parameters in `script_get_score.sh`. and then run:
```shell
cd ..    # cd to the root of program
bash script_get_score.sh
```

5. Check output in *output path*.


## Source code URL
(https://github.com/131250208/TPlinker-joint-extraction)
