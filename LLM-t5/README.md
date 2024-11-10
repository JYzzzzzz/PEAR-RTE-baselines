# t5 for CMIM23-NOM1-RA

- this program is writen by myself.


# How to use t5

-------
1. Run `convert_llm_form.py` in `../LLM-DatasetConstruction` to generate the dataset which is suitable for llm, and move the folder `CMIM23-NOM1-RA_llm_form` into `./data`.

-------
2. Prepare t5 model. "[Randeng-T5-784M-MultiTask-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese)" is recommended. Download and place it in `./models`. 

-------
3. To maintain consistency in the evaluation between baselines and PEAR, `get_score.py` need BERT ("[chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)") as tokenizer and the original CMIM23-NOM1-RA dataset as input. Place BERT in `./models` and copy the original CMIM23-NOM1-RA (the folder `CMIM23-NOM1-RA` in `../LLM-DatasetConstruction/dataset`) to `./data`. Then open `script_run.sh` and set **bert_path** and **dataset_path_for_score**.

-------
2. Configure parameters in a yaml file in path `config`. Then open `script_run.sh` and set **config_file** as the path of the yaml file.

-------
4. Run
``` commandline
bash script_run.sh
``` 
and wait until it finished.
- Process details: 
    - training and output checkpoints 
    - predict and get prediction cases of all checkpoints
    - integrate answers of llm in every checkpoint
    - get score of every checkpoint
	
-------
5. Check results in **output_dir**, which you set in yaml file.
- note: if you can't found any other results except checkpoints, open `script_run.sh` and comment out the line of `python finetune240517.py ...`. Then run
``` commandline
bash script_run.sh
```
again.

-------

