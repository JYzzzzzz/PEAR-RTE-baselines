
# Function

- The function of the program in this path is to convert the CMIM23-NOM1-RA dataset into a question-and-answer format for LLM fine-tuning.
- In the paper, the fine-tuning of T5, Qwen, and ChatGLM all used the dataset adjusted by the script in this path.

# How to use

The question-and-answer formatted dataset occupies a large storage space. Please enter the following command in the terminal to generate the dataset:
```shell
python3 convert_llm_form.py
```
After that, you can find the CMIM23-NOM1-RA dataset in `dataset/CMIM23-NOM1-RA_llm_form`, which has been converted into a question-and-answer format suitable for LLM fine-tuning.

