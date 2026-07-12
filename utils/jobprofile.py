import pandas as pd
import streamlit as st
from utils.pdf_extractor import clean_text

def build_job_components(row):
    return {
        "edu": clean_text(f"{row['degree']}, {row['major']}"),
        "exp": clean_text(f"{row['pengalaman']}, {row['tujuan']}"),
        "skill": clean_text(f"{row['kemampuan']}, {row['bahasa']}, {row['peran/tanggung_jawab']}")
    }

@st.cache_data
def load_dataset(path="data/job_profile.csv"):
    return pd.read_csv(path)

@st.cache_resource
def create_job_embeddings(df, _model):
    job_parts = df.apply(build_job_components, axis=1).tolist()
    embeddings = []
    for job in job_parts:
        embeddings.append({
            "edu": _model.encode(job['edu'], convert_to_numpy=True),
            "exp": _model.encode(job['exp'], convert_to_numpy=True),
            "skill": _model.encode(job['skill'], convert_to_numpy=True)
        })
    return embeddings