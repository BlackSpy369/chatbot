import streamlit as st
from transformers import pipeline,AutoModelForSeq2SeqLM,AutoTokenizer
import pandas as pd
from datasets import load_dataset

if "df" not in st.session_state:
    st.session_state.df=None
#sidebar
with st.sidebar:
    st.write("# 1. Input data")
    st.write("## 1.1. Use hugging face datasets module")
    dataset_name=st.text_input(label="Enter datasets Name")
    split=st.text_input(label="Enter dataset split (optional) by default is train[:10000]")
    if dataset_name:
        raw_dataset=load_dataset(dataset_name,split=split if split else "train[:10000]" )
        st.session_state.df=pd.DataFrame(raw_dataset)

    st.write("# 2. Select LLM")


#Main body

st.title("Fine Tune a LLMs")

st.subheader("Input data",divider="rainbow")
if st.session_state.df is not None:
    st.dataframe(st.session_state.df)
    