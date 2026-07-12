import re
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.parser import DEGREE_TRANSLATE_MAP

def chunk_text(text, max_len=1200):
    if len(text) <= max_len:
        return [text]
    parts = re.split(r'(?<=[.,;])\s+', text)
    chunks, current = [], ""
    for p in parts:
        if len(current) + len(p) + 1 <= max_len:
            current = f"{current} {p}".strip()
        else:
            if current: chunks.append(current)
            current = p
    if current: chunks.append(current)
    return chunks

def translate_to_english(text, tokenizer, model):
    text = str(text).strip()
    if not text:
        return text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = chunk_text(text)
    translated_chunks = []
    
    with torch.no_grad():
        for chunk in chunks:
            try:
                inputs = tokenizer([chunk], return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
                generated = model.generate(**inputs, max_length=512, num_beams=4)
                out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            except Exception as e:
                out = chunk
            translated_chunks.append(out)
    return " ".join(translated_chunks)

def compute_similarity(cv_parts, job_embeddings, model, tokenizer, trans_model, df):
    # 1. Translasi CV ke English
    tokens = str(cv_parts.get("degree", "")).split()
    trans_degree = " ".join([DEGREE_TRANSLATE_MAP.get(tok, tok) for tok in tokens])
    trans_major = translate_to_english(cv_parts.get("major", ""), tokenizer, trans_model) if cv_parts.get("major", "") else ""
    
    cv_edu_en = f"{trans_degree}, {trans_major}".strip(", ")
    cv_exp_en = translate_to_english(cv_parts["exp"], tokenizer, trans_model)
    cv_skill_en = translate_to_english(cv_parts["skill"], tokenizer, trans_model)

    # 2. Embedding CV
    cv_edu_emb = model.encode(cv_edu_en, convert_to_numpy=True)
    cv_exp_emb = model.encode(cv_exp_en, convert_to_numpy=True)
    cv_skill_emb = model.encode(cv_skill_en, convert_to_numpy=True)

    results = []
    cv_text_raw = str(cv_parts["edu"]).lower()

    for i, job_emb in enumerate(job_embeddings):
        current_job_row = df.iloc[i]
        job_major_raw = str(current_job_row["major"]).lower().strip()
        job_degree_raw = str(current_job_row["degree"]).lower().strip()

        # Aturan Khusus "Semua Jurusan" seperti di Google Colab
        if "semua jurusan" in job_major_raw:
            if job_degree_raw == "s2":
                sim_edu = 0.8 if "s2" in cv_text_raw else 0.4
            elif job_degree_raw == "s1":
                sim_edu = 0.8 if any(d in cv_text_raw for d in ["s1", "d4", "s2"]) else 0.4
            elif job_degree_raw == "d3":
                sim_edu = 0.8 if any(d in cv_text_raw for d in ["d3", "d4", "s1", "s2"]) else 0.4
            elif job_degree_raw in ["sma", "smk"]:
                sim_edu = 0.8 if any(d in cv_text_raw for d in ["sma", "smk", "d3", "d4", "s1", "s2"]) else 0.4
            elif job_degree_raw == "smp":
                sim_edu = 0.8 if any(d in cv_text_raw for d in ["smp", "sma", "smk", "d3", "d4", "s1", "s2"]) else 0.4
            else:
                sim_edu = 0.8 if job_degree_raw in cv_text_raw else 0.4
        else:
            sim_edu = cosine_similarity([cv_edu_emb], [job_emb['edu']])[0][0]

        sim_exp = cosine_similarity([cv_exp_emb], [job_emb['exp']])[0][0]
        sim_skill = cosine_similarity([cv_skill_emb], [job_emb['skill']])[0][0]

        sim_edu = max(0.0, float(sim_edu))
        sim_exp = max(0.0, float(sim_exp))
        sim_skill = max(0.0, float(sim_skill))
        
        score = (sim_skill * 0.5 + sim_exp * 0.4 + sim_edu * 0.1)

        results.append({
            "job_title": current_job_row["job_title"],
            "score": score,
            "skill": sim_skill,
            "exp": sim_exp,
            "edu": sim_edu
        })

    return pd.DataFrame(results)