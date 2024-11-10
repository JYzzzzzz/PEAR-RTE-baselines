
# guide for reproduction of qwen and chatglm

- We use **llama-factory** framework to fine-tuning qwen and chatglm.

# steps (Taking qwen as an example)

1. Prepare framework. Download [llama-factory](https://github.com/hiyouga/LLaMA-Factory).

2. Prepare dataset. Run `convert_llm_form.py` in `../LLM-DatasetConstruction` to generate the dataset which is suitable for llm, and register 3 dataset files (train, valid, test, respectively) in **llama-factory**. We give an example of register infomation in following:
```json
  "cmim23_nom1_ra_train": {
    "file_name": "my_datasets/CMIM23-NOM1-RA_llm_form/triple_order/train.json"
  },
  "cmim23_nom1_ra_dev": {
    "file_name": "my_datasets/CMIM23-NOM1-RA_llm_form/triple_order/dev.json"
  },
  "cmim23_nom1_ra_test": {
    "file_name": "my_datasets/CMIM23-NOM1-RA_llm_form/triple_order/test.json"
  },

```

3. Prepare model. download [qwen2-7b](https://huggingface.co/Qwen/Qwen2-7B) (or [glm4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) ) and place it in a suitable path.

4. Prepare scripts and config files. Move the folder `my_scripts` to the root path of **llama-factory**.

5. Prepare something for **get_score**. To maintain consistency in the evaluation between baselines and PEAR, `my_scripts/get_score.py` need BERT ("[chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)") as tokenizer and the original CMIM23-NOM1-RA dataset as input. Place BERT and the original CMIM23-NOM1-RA (the folder `CMIM23-NOM1-RA` in `../LLM-DatasetConstruction/dataset`) in suitable places. Then open `<llama-fac root>/my_scripts/run_qwen2.sh` and set **bert_path** and **dataset_path_for_score**.

6. Adjust other parameters and paths in `<llama-fac root>/my_scripts/qwen2_lora_sft.yaml` and `<llama-fac root>/my_scripts/run_qwen2.sh`. You must adjust some paths to suit your project.

7. Run
``` commandline
cd <llama-fac root>
bash my_scripts/run_qwen2.sh
```

8. Check results (`score_triple_complete.json` and `score_triple_rouge1(0.6).json`) in **output_dir** which you set in `<llama-fac root>/my_scripts/qwen2_lora_sft.yaml`.

----------

You can fine-tuning chatglm in similer steps, the differeces are:
- We use [glm4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) as pre-trained model parameters.
- We use `my_scripts/glm4_lora_sft.yaml` as the config file of chatglm.
- We use `my_scripts/run_glm4_9b.sh` as the shell script of chatglm.
