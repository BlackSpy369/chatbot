import streamlit as st
from transformers import pipeline,AutoModel,AutoTokenizer,Trainer,TrainingArguments
import pandas as pd
from datasets import load_dataset
from utils.utils import preprocess_and_tokenize,make_lora_model
from torchinfo import summary
from peft import LoraConfig


if "df" not in st.session_state:
    st.session_state.df=None

if "model" not in st.session_state:
    st.session_state.model=None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer=None

#sidebar
with st.sidebar:
    st.write("## 1. Input data")
    st.write("### 1.1. Use hugging face datasets module")
    dataset_name=st.text_input(label="Enter datasets Name",placeholder="e.g. billsum")
    split=st.text_input(label="Enter dataset split (optional) by default is train[:100]")
    if dataset_name:
        raw_dataset=load_dataset(dataset_name,split=split if split else "train[:100]" )
        st.session_state.df=pd.DataFrame(raw_dataset)

    st.write("## 2. Select LLM")
    model_name=st.selectbox(label="Enter Model name",options=["t5-small"])

    with st.expander("# 3. Training parameters"):
        train_split_ratio=st.slider(label="Train Split",min_value=10,max_value=90,value=80)
        per_device_train_batch_size=st.slider("per_device_train_batch_size",1,64,4,1)
        per_device_eval_batch_size=st.slider("per_device_eval_batch_size",1,64,4,1)
        logging_steps=st.slider("logging_steps",10,1000,100,10)
    
    use_lora=st.toggle(label="use LoRA",value=True)
    if use_lora:
        r=st.slider(label="r",min_value=2,max_value=28,value=8,step=2)
        lora_alpha=st.slider("lora_alpha",8,48,16,2)
        lora_dropout=st.slider("lora_dropout",0.0,0.5,0.05,0.01)
        st.warning("target_modules name should be seperated by comma (,)")
        target_modules=st.text_input(label="target_modules")

#Main body

st.title("Fine Tune LLMs")

if st.session_state.df is not None:

    with st.status("Running",expanded=True) as status:
        st.write("Showing Input data...")
        status.write("Splitting data...")
        raw_dataset=raw_dataset.train_test_split(train_size=train_split_ratio/100)

    st.subheader("Input data",divider="rainbow")
    st.dataframe(st.session_state.df)

    cols=st.columns(3)
    with cols[0]:
        st.write("**Training Samples**")
        st.write(str(len(raw_dataset["train"])))
    with cols[1]:
        st.write("**Test Samples**")
        st.write(str(len(raw_dataset["test"])))
    with cols[2]:
        st.write("**Train Ratio**")
        st.write(str(train_split_ratio)+"%")

    cols=st.columns(2)
    with cols[0]:#Input
        X_col_name=st.selectbox(label="Select Input (X)",options=st.session_state.df.columns)
    with cols[1]:#Label
        y_col_name=st.selectbox(label="Select target (y)",options=st.session_state.df.columns)

    # if st.button("Fine Tune LLM"):
    if X_col_name==y_col_name:#Checking whether X and y are same
        st.error("Input(X) and target(y) cannot be same...")
        status.update(label="An error occured",state="error")
        st.stop()

    status.write("Downloading model...")
    if st.session_state.model==None:
        st.session_state.model=AutoModel.from_pretrained(model_name)

    status.write("Downloading tokenizer...")
    if st.session_state.tokenizer==None:
        st.session_state.tokenizer=AutoTokenizer.from_pretrained(model_name)

    status.write("Tokenizing data...")

    tokenized_dataset=raw_dataset.map(preprocess_and_tokenize,fn_kwargs={"X_col_name":X_col_name,"y_col_name":y_col_name,"tokenizer":st.session_state.tokenizer},batched=True)
    # print(tokenized_dataset[0])
    
    with st.container(height=500):
        st.subheader("Model Summary:",divider="rainbow")
        cols=st.columns(2)
        with cols[0]:
            st.write(summary(st.session_state.model))
        with cols[1]:
            st.write(str(st.session_state.model))
    
    if use_lora:
        status.write("Applying LoRA...")
        st.session_state.lora_config=LoraConfig(r=r,lora_alpha=lora_alpha,lora_dropout=lora_dropout,target_modules=target_modules.split(","))
        st.session_state.lora_model=make_lora_model(st.session_state.model,st.session_state.lora_config)
        with st.container(height=500):
            st.subheader("LoRA Model Summary:",divider="rainbow")
            cols=st.columns(2)
            with cols[0]:
                st.write(summary(st.session_state.lora_model))
            with cols[1]:
                st.write(st.session_state.lora_model)

    status.write("Fine Tuning LLM...")
    training_args=TrainingArguments(
        output_dir="output",
        per_device_eval_batch_size=per_device_eval_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps
    )

    trainer=Trainer(
        st.session_state.lora_model if st.session_state.lora_model else st.session_state.model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=st.session_state.tokenizer
    )

    trainer.train()
    st.write("### Model trained")