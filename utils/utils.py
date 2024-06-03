import numpy as np
from transformers import AutoTokenizer

def preprocess_and_tokenize(examples,X_col_name:str,y_col_name:str,tokenizer:AutoTokenizer):
    inputs=tokenizer(examples[X_col_name])
    outputs=tokenizer(examples[y_col_name])

    inputs["labels"]=outputs["input_ids"]

    return inputs