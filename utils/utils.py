from transformers import AutoTokenizer
from peft import LoraConfig,get_peft_model

def preprocess_and_tokenize(examples,X_col_name:str,y_col_name:str,tokenizer:AutoTokenizer):
    inputs=tokenizer(examples[X_col_name])
    outputs=tokenizer(examples[y_col_name])

    inputs["labels"]=outputs["input_ids"]

    return inputs

def make_lora_model(model,lora_config:LoraConfig,adapter_name:str="default"):
    lora_model=get_peft_model(model,lora_config,adapter_name=adapter_name)
    return lora_model