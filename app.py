import streamlit as st
import pandas as pd
import os
import urllib.parse
import altair as alt

from utils.model import load_model
from utils.pdf_extractor import extract_text_from_pdf
from utils.parser import parse_cv_structured
from utils.similarity import compute_similarity
from utils.feedback import generate_feedback
from utils.jobprofile import load_dataset, create_job_embeddings
from utils.model import load_model, load_translation_model

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

@st.cache_resource
def get_translation_model():
    return load_translation_model()

translation_tokenizer, translation_model = get_translation_model()

df = load_dataset()
job_embeddings = create_job_embeddings(df, model)

# ======================
# SAVE FUNCTION
# ======================
def save_data(file_path, new_row, key_cols):

    new_df = pd.DataFrame([new_row])

    if not os.path.exists(file_path):
        new_df.to_csv(file_path, index=False)
        return

    df_save = pd.read_csv(file_path)

    for col in new_df.columns:
        if col not in df_save.columns:
            df_save[col] = None

    for col in df_save.columns:
        if col not in new_df.columns:
            new_df[col] = None

    new_df = new_df[df_save.columns]

    for col in key_cols:
        df_save[col] = df_save[col].astype(str)
        new_df[col] = new_df[col].astype(str)

    mask = pd.Series([True] * len(df_save))
    for col in key_cols:
        mask &= (df_save[col] == new_df.iloc[0][col])

    if mask.any():
        df_save.loc[mask, :] = new_df.iloc[0].values
    else:
        df_save = pd.concat([df_save, new_df], ignore_index=True)

    df_save.to_csv(file_path, index=False)

def style_header(df):
    return df.style.set_table_styles([
        # Header: Hijau Kalla, Putih, Tengah
        {'selector': 'th', 'props': [
            ('background-color', '#0B5334'), 
            ('color', 'white'),
            ('font-family', 'sans-serif'),
            ('text-align', 'center !important'),
            ('vertical-align', 'middle !important'),
            ('padding', '15px')
        ]},
        # Data: Tengah, Padding rapi
        {'selector': 'td', 'props': [
            ('font-family', 'sans-serif'),
            ('padding', '10px'),
            ('text-align', 'center !important'),
            ('vertical-align', 'middle !important')
        ]}
    ]).set_properties(**{
        'background-color': 'white',
        'color': '#333',
        'border-color': '#f0f0f0',
        'text-align': 'center !important'
    })
# ======================
# CONFIG
# ======================
st.set_page_config(layout="wide", page_title="Dashboard Penilaian CV")

# ======================
# CSS DESIGN
# ======================
st.markdown("""
<style>
.stApp { background-color: #F8F9FA; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}

.main-title {
    color: #0B5334;
    font-weight: 700;
    font-size: 24px;
}

.custom-card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #F0F0F0;
    
    /* KUNCI UTAMA: Berikan tinggi minimal yang cukup */
    min-height: 220px; 
    
    /* Pakai Flexbox agar konten bisa diatur jaraknya */
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}

.custom-card1 {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #F0F0F0;
    margin-bottom: 20px;
}

.custom-progress {
    height: 10px;
    background-color: #E9ECEF;
    border-radius: 5px;
}

.progress-fill {
    height: 100%;
    background-color: #28A745;
    border-radius: 5px;
}

.badge-match {
    background-color: #28A745;
    color: white;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 14px;
    font-weight: bold;
}

.card-best-match {
    background-color: #FFF9E6;
    border: 1px solid #FFE082;
    padding: 15px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    margin-top:15px;
}

/* --- STYLE NAVIGASI TAB (GAMBAR 2) --- */
.tab-container {
    display: flex;
    justify-content: center;
    width: 100%;
    background-color: white;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 25px;
}

.tab-button {
    flex: 1;
    padding: 12px;
    text-align: center;
    text-decoration: none;
    color: #0B5334;
    font-weight: 600;
    transition: 0.3s;
    border-right: 1px solid #E0E0E0;
    background: none;
    border-top: none;
    border-bottom: none;
    border-left: none;
    cursor: pointer;
}

.tab-button:last-child {
    border-right: none;
}

.tab-active {
    background-color: #0B5334 !important;
    color: white !important;
}
/* Menghapus warna primary bawaan dan mengganti ke hijau muda custom */
button[kind="primary"] {
    background-color: #bceba2 !important;
    color: #0B5334 !important; /* Tulisan diganti hijau gelap agar kontras */
    border: none !important;
    font-size: 25px !important;   /* Ukuran font lebih besar */
    font-weight: 800 !important;
}

/* Efek saat kursor menempel (Hover) */
button[kind="primary"]:hover {
    background-color: #a8d691 !important;
    border: none !important;
    font-size: 25px !important;   /* Ukuran font lebih besar */
    font-weight: 800 !important;
}

/* Merapikan label Selectbox di Sidebar */
[data-testid="stSidebar"] label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #444 !important;
    margin-bottom: 8px !important;
}

/* Memberikan efek fokus hijau pada input sidebar */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1px solid #E0E0E0 !important;
}
/* Merapikan inputan di sidebar agar garisnya jelas */
[data-testid="stSidebar"] input[type="text"] {
    border: 2px solid #E0E0E0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input[type="text"]:focus {
    border-color: #0B5334 !important;
}
[data-testid="stSidebar"] .stTextInput {
    margin-bottom: -20px !important;
}
.block-container {
            padding-top: 1.5rem !important;            
</style>
""", unsafe_allow_html=True)

def get_score_color(val):
    if val >= 50:
        return "#28A745"   # hijau
    elif val >= 35:
        return "#FFC107"   # kuning
    else:
        return "#DC3545"   # merah

# ======================
# MENU
# ======================
# menu = st.radio("", ["Single CV", "Leaderboard"], horizontal=True)
st.markdown("""
    <div style="
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #0B5334;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <h1 style="
                margin: 0px;
                color: #0B5334;
                font-size: 30px;
                font-weight: 800;
            ">
                Dashboard Penilaian CV
            </h1>
            <p style="
                margin: 4px 0 0 0;
                color: #555;
                font-size: 15px;
            ">
                Sistem ini membantu proses seleksi kandidat dengan membandingkan isi CV 
                terhadap kebutuhan posisi di <span style="color: #0B5334; font-weight: 600;">Kalla Aspal</span> secara lebih terstruktur. 
                Hasil analisis ditampilkan dalam bentuk skor, ranking, dan Rekomendasi Job/Posisi yang paling terbaik 
                berbasis <i>Sentence-BERT</i>
            </p>
        </div>
        
    </div>
""", unsafe_allow_html=True)

# --- LOGIKA NAVIGASI TAB ---
if 'menu' not in st.session_state:
    st.session_state.menu = "Single CV"

# Membuat Tab Layout
cols = st.columns([1, 1, 1])

with cols[0]:
    active_class = "tab-active" if st.session_state.menu == "Single CV" else ""
    if st.button("Single CV", use_container_width=True, type="primary" if st.session_state.menu == "Single CV" else "secondary"):
        st.session_state.menu = "Single CV"
        st.rerun()

with cols[1]:
    if st.button("Leaderboard", use_container_width=True, type="primary" if st.session_state.menu == "Leaderboard" else "secondary"):
        st.session_state.menu = "Leaderboard"
        st.rerun()

with cols[2]:
    if st.button("Bulk CV", use_container_width=True, type="primary" if st.session_state.menu == "Bulk CV" else "secondary"):
        st.session_state.menu = "Bulk CV"
        st.rerun()

# Simpan variabel menu agar kode di bawahnya tetap jalan
menu = st.session_state.menu

# =========================================================
# SINGLE CV
# =========================================================
if menu == "Single CV":
    with st.sidebar:
        st.markdown('<p class="main-title">📥 Input Data Kandidat</p>', unsafe_allow_html=True)
        
        # --- SEKSI INFORMASI DASAR ---
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 5px solid #0B5334; margin-bottom: 5px;'>
                <p style='margin:0; font-size: 12px; color: gray; font-weight: bold;'>👤 INFORMASI DASAR</p>
            </div>
        """, unsafe_allow_html=True)
        
        nama = st.text_input("Nama Lengkap", placeholder="Ketik nama pelamar...", label_visibility="collapsed")
        hp = st.text_input("Nomor WhatsApp", placeholder="Contoh: 62812...", label_visibility="collapsed")
        
        # Jeda antar seksi
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

        # --- SEKSI TARGET POSISI ---
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 5px solid #0B5334; margin-bottom: -20px;'>
                <p style='margin:0; font-size: 12px; color: gray; font-weight: bold;'>🎯 TARGET POSISI</p>
            </div>
        """, unsafe_allow_html=True)
        selected_job = st.selectbox("", df['job_title'], key="single_job_select")
        
        # --- UPLOAD & TOMBOL ---
        file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
        process_btn = st.button("🚀 Proses Screening", use_container_width=True, type="primary")

    # --- LOGIKA VALIDASI ---
    if process_btn and not file:
        st.warning("⚠️ Silakan upload file CV terlebih dahulu.")
        st.stop()

    if file and process_btn:
        if not nama:
            st.error("⚠️ Mohon isi Nama Lengkap pelamar.")
            st.stop()

        with st.spinner("⏳⏳⏳ Menganalisis CV..."):
            # Proses Backend
            text = extract_text_from_pdf(file)
            cv_extracted = parse_cv_structured(text)
            # df_sim = compute_similarity(cv_extracted,job_embeddings,model,df)
            df_sim = compute_similarity(cv_extracted, job_embeddings, model,translation_tokenizer, translation_model, df)
            df_rank = df_sim.sort_values(by="score", ascending=False)

            # Data untuk Job yang dipilih
            selected_row = df_rank[df_rank['job_title'] == selected_job].iloc[0]
            score       = max(0.0, selected_row['score'] * 100)
            exp_score   = max(0.0, selected_row['exp']   * 100)
            edu_score   = max(0.0, selected_row['edu']   * 100)
            skill_score = max(0.0, selected_row['skill'] * 100)

            best_global   = df_rank.iloc[0]
            best_job_name  = best_global['job_title']
            best_job_score = max(0.0, best_global['score'] * 100)

        # --- TAMPILAN DASHBOARD UTAMA ---
        st.markdown(f"<p class='main-title' style='font-size: 40px; margin-bottom: 0px;'>Dashboard Penilaian CV: {nama}</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            score_color = get_score_color(score)
            st.markdown(f"""
            <div class="custom-card">
                <p style="color: gray; font-weight: bold; text-align:center; margin-bottom: 0px;">Skor Kecocokan</p>
                <h1 style="color: {score_color}; font-size: 58px; margin-top: -10px; margin-bottom: 10px; text-align:center;">{score:.1f}%</h1>
                <div class="custom-progress"><div class="progress-fill" style="width:{score}%; background-color:{score_color};"></div></div>
                <div style="flex-grow: 1; display: flex; align-items: flex-end; margin-top: 20px;">
                    <p style="background: #F1F3F4; padding: 12px; border-radius: 8px; width: 100%; text-align: center; font-size: 18px; margin: 0; color: #333;">
                        <b>JOB:</b> {selected_job}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            def render_bar(val):
                color = get_score_color(val)
                return f"""
                    <div style="height:10px; background-color:#E9ECEF; border-radius:5px; overflow:hidden;">
                        <div style="width:{val:.1f}%; height:100%; background-color:{color}; border-radius:5px; transition: width 0.3s;"></div>
                    </div>
                """
            st.markdown(f"""
            <div class="custom-card">
                <p style="color: gray; font-weight: bold; margin-bottom:20px;">Detail Penilaian</p>
                <p style="margin-bottom:4px; font-size:16px; color: #444;">
                    🎓 Pendidikan
                    <span style="float:right; font-weight:bold; color:{get_score_color(edu_score)};">
                        {edu_score:.1f}%
                    </span>
                </p>
                <div style="margin-bottom:15px;">{render_bar(edu_score)}</div>
                <p style="margin-bottom:4px; font-size:16px; color: #444;">
                    💼 Pengalaman
                    <span style="float:right; font-weight:bold; color:{get_score_color(exp_score)};">
                        {exp_score:.1f}%
                    </span>
                </p>
                <div style="margin-bottom:15px;">{render_bar(exp_score)}</div>
                <p style="margin-bottom:4px; font-size:16px; color: #444;">
                    ⭐ Kemampuan
                    <span style="float:right; font-weight:bold; color:{get_score_color(skill_score)};">
                        {skill_score:.1f}%
                    </span>
                </p>
                <div>{render_bar(skill_score)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="custom-card">
                <p style="color: gray; font-weight: bold; margin-bottom: 20px;">Informasi Pendidikan</p>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #f1f1f1; padding-bottom: 8px;">
                    <p style="font-size: 16px; color: gray; margin: 0;">🎓 Tingkat</p>
                    <p style="font-size: 19px; font-weight: bold; margin: 0; color: #333;">{str(cv_extracted['degree']).upper()}</p>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <p style="font-size: 16px; color: gray; margin: 0;">📖 Program Studi</p>
                    <p style="font-size: 19px; font-weight: bold; margin: 0; color: #333; text-align: right;">{str(cv_extracted['major']).title()}</p>
                </div>
                <div style="flex-grow: 1;"></div>
            </div>
            """, unsafe_allow_html=True)

        # --- SEKSI BEST MATCH & WHATSAPP ---
        colA, colB = st.columns([2, 1])
        with colA:
            best_match_color = get_score_color(best_job_score)
            st.markdown(f"""
            <div class="card-best-match">
                <div style="font-size: 30px; margin-right: 15px;">🏆</div>
                <div>
                    <span style="color: #28A745; font-weight: bold; font-size: 24px;">Best Job Match</span>
                    <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
                        <span style="font-weight: bold; font-size: 28px; color: #333;">{best_job_name}</span>
                        <span class="badge-match" style="background-color:{best_match_color};">{best_job_score:.1f}% Match</span>
                    </div>
                    <p style="font-size: 13px; color: gray; margin-bottom: 15px;">Ini adalah Best Job Untuk CV.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            if hp:
                nomor_wa = hp.replace("+","").replace(" ","")
                pesan_wa = urllib.parse.quote(f"Selamat {nama}, Anda Berhasil Untuk Melanjutkan Ke Tahap Selanjutnya, dengan Skor CV {score:.2f}% untuk posisi {selected_job}. Terimakasih sudah Melakukan Pendaftaran. Untuk Informasi Tahap Selanjutnya Akan Dihubungi Melalui Whatsapp Ini!")
                st.markdown(f"""
                <div style="background: white; border: 1px solid #F0F0F0; padding: 15px; margin-top:15px; border-radius: 12px;">
                    <p style="font-weight: bold; color: #333; margin-bottom: 5px; font-size:22px;">Rekomendasi</p>
                    <p style="font-size: 13px; color: gray; margin-bottom: 15px;">Kirim hasil rekomendasi ini ke WhatsApp untuk ditinjau lebih lanjut.</p>
                    <a href="https://wa.me/{nomor_wa}?text={pesan_wa}" target="_blank" style="text-decoration: none;">
                        <button style="width: 100%; background-color: #28A745; color: white; padding: 10px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; display:flex; align-items:center; justify-content:center; gap:8px;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
                                <path d="M12.04 2C6.58 2 2.13 6.45 2.13 11.91c0 1.75.46 3.46 1.32 4.97L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21h.01c5.46 0 9.9-4.45 9.9-9.91 0-2.65-1.03-5.14-2.9-7.01A9.876 9.876 0 0012.04 2zm5.84 14.07c-.24.69-1.41 1.32-1.95 1.4-.5.08-1.13.11-1.83-.12-.42-.13-.96-.31-1.65-.61-2.91-1.26-4.81-4.18-4.96-4.38-.14-.2-1.18-1.57-1.18-3 0-1.43.75-2.13 1.02-2.42.26-.29.57-.36.76-.36.19 0 .38 0 .54.01.18.01.41-.07.64.49.24.58.81 2 .88 2.14.07.14.12.31.02.5-.1.19-.15.31-.29.48-.14.17-.3.37-.43.5-.14.14-.29.29-.13.57.17.29.74 1.22 1.59 1.97 1.09.97 2.01 1.27 2.3 1.41.29.14.46.12.63-.07.17-.19.72-.84.91-1.13.19-.29.38-.24.64-.14.26.1 1.66.78 1.94.93.29.14.48.21.55.33.07.12.07.69-.17 1.38z"/>
                            </svg>
                            Kirim WhatsApp
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
        
        
        # --- TABEL TOP 5 ---
        # st.markdown("### 📊 Top 5 Job")
        st.markdown("""
        <h3 style="
            color:#0B5334;
            font-size:22px;
            font-weight:700;
            margin-bottom:10px;
        ">
        📊 Top 5 Job
        </h3>
        """, unsafe_allow_html=True)
        top5 = df_rank.head(5).copy().reset_index(drop=True)
        top5["Rank"]       = top5.index + 1
        top5["Score"]      = top5["score"].apply(lambda x: f"{max(0.0, x*100):.2f}%")
        top5["Skill"]      = top5["skill"].apply(lambda x: f"{max(0.0, x*100):.2f}%")
        top5["Experience"] = top5["exp"].apply(lambda x: f"{max(0.0, x*100):.2f}%")
        top5["Education"]  = top5["edu"].apply(lambda x: f"{max(0.0, x*100):.2f}%")
        
        df_to_show = top5[["Rank", "job_title", "Score", "Skill", "Experience", "Education"]]
        df_to_show = df_to_show.rename(columns={
            "job_title": "Posisi",
            "Score": "Skor CV",
            "Skill": "Kemampuan",
            "Experience": "Pengalaman",
            "Education": "Pendidikan"
        })

        st.table(style_header(df_to_show))
        # st.table(style_header(df_to_show))

        # --- SAVE DATA SEMUA JOB KE DATABASE ---
        # Kita melakukan perulangan untuk setiap baris di df_rank 
        # (df_rank berisi skor pelamar ini untuk SELURUH posisi job yang ada di database)
        
        # --- SAVE DATA TOP 5 JOB KE DATABASE ---
        # Kita ambil hanya 5 posisi dengan skor tertinggi untuk pelamar ini
        df_top5_save = df_rank.head(5)

        # st.markdown("### 🧠 Analisis & Saran Pengembangan")
        st.markdown("""
        <h3 style="
            color:#0B5334;
            font-size:22px;
            font-weight:700;
            margin-bottom:10px;
        ">
        🧠 Analisis & Saran Pengembangan
        </h3>
        """, unsafe_allow_html=True)
        isi_feedback = generate_feedback(
            cv_parts=cv_extracted, 
            best_job=selected_job, 
            score=float(score),
            skill=float(skill_score)/100,
            exp=float(exp_score)/100,
            edu=float(edu_score)/100
        )
        # Fix newline
        isi_feedback = isi_feedback.replace("\n", "<br>")
        warna_border = "#28A745" if score >= 65 else "#FFC107" if score >= 45 else "#DC3545"
        st.markdown(f"""
        <div style="
            background: #ffffff;
            padding: 25px;
            border-radius: 18px;
            border: 1px solid #eaeaea;
            border-left: 6px solid {warna_border};
            box-shadow: 0 8px 25px rgba(0,0,0,0.04);
            line-height: 1.8;
            max-width: 100%;
            width:100%;
        ">
            <div style="
                color: #444;
                font-size: 15px;
                text-align: justify;
                font-family: 'Segoe UI', sans-serif;
                margin-top:-30px;
            ">
                {isi_feedback}
            </div>
            <div style="
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px dashed #ddd;
                font-size: 12px;
                color: #888;
            ">
                Hasil ini dihasilkan berdasarkan analisis otomatis dan dapat digunakan sebagai referensi awal dalam proses seleksi.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner(f"💾 Menyimpan Top 5 rekomendasi untuk {nama}..."):
            for index, row_job in df_top5_save.iterrows():
                data_to_save = {
                    "nama": nama,
                    "hp": hp,
                    "job": row_job['job_title'],
                    "rekomendasi_ai": best_job_name,
                    "score": max(0.0, round(row_job['score'] * 100, 2)),
                    "skill": max(0.0, round(row_job['skill'] * 100, 2)),
                    "exp":   max(0.0, round(row_job['exp']   * 100, 2)),
                    "edu":   max(0.0, round(row_job['edu']   * 100, 2))
                }
                save_data("leaderboard.csv", data_to_save, ["nama", "job", "hp","rekomendasi_ai","score","skill","exp","edu",])
        
        st.success(f"✅ Berhasil menyimpan Top 5 posisi terbaik untuk {nama}!")

    elif not file:
    # --- Tampilan Welcome / Tutorial ---
        st.markdown("""
            <h3 style="color:#0B5334; margin-bottom:-10px;">
            🚀 Selamat Datang di Dashboard Penilaian CV
            </h3>
            """, unsafe_allow_html=True)
        st.markdown("""
            <p style="color:#0B5334; font-size:16px;">
            Aplikasi ini membantu Anda menyeleksi kandidat terbaik menggunakan AI dengan cepat dan akurat. 
            Berikut adalah panduan fitur utamanya:
            </p>
            """, unsafe_allow_html=True)
                    
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background-color: #f0f7f4; padding: 20px; border-radius: 10px; border-left: 5px solid #0B5334; height: 100%;">
                <h4 style="color: #0B5334; margin-top:0;">📄 Single CV</h4>
                <p style="font-size: 14px; color: #444;">
                    <b>Single CV</b> digunakan untuk menganalisis <b>satu kandidat secara lebih detail</b>. 
                    Fitur ini menampilkan skor kecocokan CV dengan posisi yang dilamar, 
                    serta memberikan rekomendasi job yang lain. Cocok digunakan ketika ingin mengevaluasi satu kandidat secara lebih fokus.
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(" ", unsafe_allow_html=True)    
        with col2:
            st.markdown("""
            <div style="background-color: #fdfaf0; padding: 20px; border-radius: 10px; border-left: 5px solid #f39c12; height: 100%;">
                <h4 style="color: #f39c12; margin-top:0;">🏆 Leaderboard</h4>
                <p style="font-size: 14px; color: #444;">
                    <b>Leaderboard</b> merupakan halaman <b>peringkat kandidat berdasarkan skor</b>. 
                    Semua CV yang sudah dianalisis akan ditampilkan secara terurut dari yang tertinggi, 
                    serta memudahkan HR dalam membandingkan performa masing-masing kandidat. 
                    Cocok digunakan untuk membantu proses seleksi secara lebih cepat.
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(" ", unsafe_allow_html=True)    
        with col3:
            st.markdown("""
            <div style="background-color: #f0f4f7; padding: 20px; border-radius: 10px; border-left: 5px solid #2980b9; height: 100%;">
                <h4 style="color: #2980b9; margin-top:0;">📁 Bulk CV</h4>
                <p style="font-size: 14px; color: #444;">
                    <b>Bulk CV</b> digunakan untuk menangani <b>banyak file CV sekaligus</b>. 
                    Anda dapat mengunggah beberapa file dalam satu waktu tanpa harus satu per satu, 
                    lalu sistem akan memproses dan menampilkan skor CV secara otomatis, serta memberikan Rekomendasi job yang paling Terbaik.
                    Cocok digunakan saat menghadapi jumlah pelamar yang cukup banyak.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(" ", unsafe_allow_html=True)
            
        st.info("👈 **Mulai Sekarang:** Silakan pilih posisi pekerjaan dan upload file CV Anda melalui sidebar di sebelah kiri.")
# LEADERBOARD
# =========================================================
elif menu == "Leaderboard":
    def style_leaderboard_custom(df):
        return df.style.set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#0B5334'), ('color', 'white'),
                ('text-align', 'center !important'), ('padding', '15px')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center !important'), ('padding', '12px')
            ]}
        ]).set_properties(**{'background-color': 'white', 'color': '#333', 'text-align': 'center !important'})
    with st.sidebar:
        st.markdown('<p class="main-title">🔍 Filter Leaderboard</p>', unsafe_allow_html=True)
        if "lb_mode" not in st.session_state:
            st.session_state.lb_mode = "Single CV"
        st.markdown("<p style='font-size:13px; font-weight:bold; margin-bottom:5px; color:#444;'>Pilih Sumber Data:</p>", unsafe_allow_html=True)
        col_lb1, col_lb2 = st.columns(2)
        with col_lb1:
            if st.button("Single CV", use_container_width=True, type="primary" if st.session_state.lb_mode == "Single CV" else "secondary"):
                st.session_state.lb_mode = "Single CV"
                st.rerun()
        with col_lb2:
            if st.button("Bulk CV", use_container_width=True, type="primary" if st.session_state.lb_mode == "Bulk CV" else "secondary"):
                st.session_state.lb_mode = "Bulk CV"
                st.rerun()
        lb_mode = st.session_state.lb_mode
        file_path = "leaderboard.csv" if lb_mode == "Single CV" else "leaderboardbulk.csv"
        if not os.path.exists(file_path):
            st.error(f"📂 Belum ada data {lb_mode}.")
            st.stop()
        df_lb = pd.read_csv(file_path)
        job_col = "job" if "job" in df_lb.columns else "job_input" if "job_input" in df_lb.columns else None
        name_col = "nama" if "nama" in df_lb.columns else "cv"
        hp_col = "hp" if "hp" in df_lb.columns else None
        if job_col:
            st.markdown("<div style='background-color:#f8f9fa; padding:10px; border-radius:10px; border-left:5px solid #0B5334; margin-bottom:-20px;'><p style='margin:0; font-size:11px; color:gray; font-weight:bold;'>🎯 FILTER POSISI</p></div>", unsafe_allow_html=True)
            job_list = ["All"] + sorted(df_lb[job_col].dropna().unique().tolist())
            selected_job = st.selectbox("", job_list, key="lb_job_filter")
        else:
            selected_job = "All"
        st.markdown("<div style='background-color:#f8f9fa; padding:10px; border-radius:10px; border-left:5px solid #0B5334; margin-bottom:-20px; margin-top:10px;'><p style='margin:0; font-size:11px; color:gray; font-weight:bold;'>🏆 TAMPILKAN PERINGKAT</p></div>", unsafe_allow_html=True)
        top_n = st.selectbox("", [1, 3, 5, 10, 20], index=1, key="lb_top_n")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        total_kandidat_unik = df_lb[name_col].nunique()
        st.markdown(f"""
        <div class="sidebar-stats-card">
            <div style="display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-bottom: 0px;">
                <span style="color: #0B5334; font-weight: bold; font-size: 18px; ">Total Kandidat</span>
            </div>
            <h2 style="color: #0B5334; margin: 0px; padding: 0px; font-size: 28px; line-height: 1.2;">
                {total_kandidat_unik} <small style="font-size: 14px; color: gray; font-weight: normal;">Orang</small>
            </h2>
            <p style="color: gray; font-size: 10px; margin: 0px; padding: 0px; margin-top: -2px;">
                Terdeteksi dari total {len(df_lb)} entri posisi
            </p>
        </div>
        """, unsafe_allow_html=True)
    # 2. --- LOGIKA FILTERING ---
    df_filtered = df_lb.copy()
    if job_col and selected_job != "All":
        df_filtered = df_filtered[df_filtered[job_col] == selected_job]
    if "score" in df_filtered.columns:
        df_filtered["score"] = pd.to_numeric(df_filtered["score"].astype(str).str.replace('%', ''), errors='coerce')
        df_filtered = df_filtered.sort_values(by="score", ascending=False)
    for col in ["score", "skill", "exp", "edu"]:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].apply(lambda x: max(0.0, float(x)) if pd.notnull(x) else 0.0)
    df_filtered = df_filtered.head(top_n).reset_index(drop=True)
    df_filtered["rank"] = df_filtered.index + 1
    # 3. --- TAMPILAN UTAMA ---
    st.markdown(f'<p class="main-title">🏆 Leaderboard CV – {lb_mode}</p>', unsafe_allow_html=True)
    st.info(f"📍 Menampilkan Top {len(df_filtered)} kandidat untuk lowongan: **{selected_job}**")
    if not df_filtered.empty:
        # --- GRAFIK ---
        st.markdown("""
            <h3 style="color:#0B5334; font-size:22px; font-weight:700; margin-bottom:10px;">
            📈 Grafik Skor (%)
            </h3>
            """, unsafe_allow_html=True)
        df_chart = df_filtered.copy()
        if df_chart["score"].max() <= 1.0:
            df_chart["score"] = df_chart["score"] * 100
        chart = alt.Chart(df_chart).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color='#28A745').encode(
            x=alt.X(f"{name_col}:N", sort='-y', title="Nama CV"),
            y=alt.Y("score:Q", title="Skor (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[name_col, "score"]
        ).properties(height=400)
        text = chart.mark_text(baseline='bottom', dy=-5, fontWeight='bold').encode(text=alt.Text('score:Q', format='.2f'))
        st.altair_chart(chart + text, use_container_width=True)
        # --- TABEL ---
        # --- TABEL DENGAN TOMBOL WHATSAPP DI DALAM KOLOM ---
        st.markdown("""
            <h3 style="color:#0B5334; font-size:22px; font-weight:700; margin-bottom:10px;">
            📋 Detail Tabel Peringkat
            </h3>
            """, unsafe_allow_html=True)
        df_table = df_filtered.copy()
        cols_format = ["score", "skill", "exp", "edu"]
        for col in cols_format:
            if col in df_table.columns:
                df_table[col] = df_table[col].apply(lambda x: f"{float(x):.2f}%" if pd.notnull(x) else "-")
        has_wa = hp_col and hp_col in df_filtered.columns
        # Header tabel
        header_cols = ["Rank", "Nama CV", "Posisi", "Skor CV", "Kemampuan", "Pengalaman", "Pendidikan"]
        if has_wa:
            header_cols.append("WhatsApp")
        header_html = "".join([f"<th style='padding:14px; text-align:center;'>{h}</th>" for h in header_cols])
        rows_html = ""
        for _, row in df_table.iterrows():
            rank_val   = row.get("rank", "-")
            nama_val   = row.get(name_col, "-")
            job_val    = row.get(job_col, "-") if job_col else "-"
            score_val  = row.get("score", "-")
            skill_val  = row.get("skill", "-")
            exp_val    = row.get("exp", "-")
            edu_val    = row.get("edu", "-")
            cells = f"""
                <td style="padding:12px; text-align:center;">{rank_val}</td>
                <td style="padding:12px; text-align:center;">{nama_val}</td>
                <td style="padding:12px; text-align:center;">{job_val}</td>
                <td style="padding:12px; text-align:center;">{score_val}</td>
                <td style="padding:12px; text-align:center;">{skill_val}</td>
                <td style="padding:12px; text-align:center;">{exp_val}</td>
                <td style="padding:12px; text-align:center;">{edu_val}</td>
            """
            if has_wa:
                hp_val = str(row.get(hp_col, "")).strip()
                if hp_val and hp_val.lower() != "nan":
                    nomor_wa = hp_val.replace("+", "").replace(" ", "")
                    raw_score = df_filtered.loc[row.name, "score"] if "score" in df_filtered.columns else 0
                    pesan_wa = urllib.parse.quote(
                        f"Selamat {nama_val}, Anda Berhasil Untuk Melanjutkan Ke Tahap Selanjutnya, dengan Skor CV {raw_score:.2f}% untuk posisi {job_val}. Terimakasih sudah Melakukan Pendaftaran. Untuk Informasi Tahap Selanjutnya Akan Dihubungi Melalui Whatsapp Ini!"
                    )
                    wa_button = f"""
                        <a href="https://wa.me/{nomor_wa}?text={pesan_wa}" target="_blank" style="text-decoration:none;">
                            <button style="background-color:#25D366; border:none; border-radius:8px; padding:8px 12px; cursor:pointer; display:flex; align-items:center; justify-content:center; margin:0 auto;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
                                    <path d="M12.04 2C6.58 2 2.13 6.45 2.13 11.91c0 1.75.46 3.46 1.32 4.97L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21h.01c5.46 0 9.9-4.45 9.9-9.91 0-2.65-1.03-5.14-2.9-7.01A9.876 9.876 0 0012.04 2zm5.84 14.07c-.24.69-1.41 1.32-1.95 1.4-.5.08-1.13.11-1.83-.12-.42-.13-.96-.31-1.65-.61-2.91-1.26-4.81-4.18-4.96-4.38-.14-.2-1.18-1.57-1.18-3 0-1.43.75-2.13 1.02-2.42.26-.29.57-.36.76-.36.19 0 .38 0 .54.01.18.01.41-.07.64.49.24.58.81 2 .88 2.14.07.14.12.31.02.5-.1.19-.15.31-.29.48-.14.17-.3.37-.43.5-.14.14-.29.29-.13.57.17.29.74 1.22 1.59 1.97 1.09.97 2.01 1.27 2.3 1.41.29.14.46.12.63-.07.17-.19.72-.84.91-1.13.19-.29.38-.24.64-.14.26.1 1.66.78 1.94.93.29.14.48.21.55.33.07.12.07.69-.17 1.38z"/>
                                </svg>
                            </button>
                        </a>
                    """
                else:
                    wa_button = "<span style='color:#ccc;'>-</span>"
                cells += f'<td style="padding:12px; text-align:center;">{wa_button}</td>'
            rows_html += f"<tr style='border-bottom:1px solid #f0f0f0;'>{cells}</tr>"
        table_html = f"""
        <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; background:white; border-radius:10px; overflow:hidden;">
            <thead>
                <tr style="background-color:#0B5334; color:white;">
                    {header_html}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada data untuk ditampilkan.")

    
# =========================================================
# BULK CV (FINAL + RESET FILE UPLOAD)
elif menu == "Bulk CV":
    # --- SIDEBAR BULK INPUT ---
    # --- SIDEBAR BULK INPUT (VERSI TOMBOL BIASA) ---
    with st.sidebar:
        st.markdown('<p class="main-title">📥 Upload CV Kandidat</p>', unsafe_allow_html=True)
        
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        if "bulk_mode" not in st.session_state:
            st.session_state.bulk_mode = "Otomatis"

        # --- TOMBOL MODE (Otomatis / Manual) ---
        st.markdown("<p style='font-size:13px; font-weight:bold; margin-bottom:5px; color:#444;'>Mode Analisis:</p>", unsafe_allow_html=True)
        col_mode1, col_mode2 = st.columns(2)
        
        with col_mode1:
            # Jika mode adalah Otomatis, tombol jadi warna hijau (primary)
            if st.button("Otomatis", use_container_width=True, type="primary" if st.session_state.bulk_mode == "Otomatis" else "secondary"):
                st.session_state.bulk_mode = "Otomatis"
                st.rerun()
                
        with col_mode2:
            # Jika mode adalah Manual, tombol jadi warna hijau (primary)
            if st.button("Manual", use_container_width=True, type="primary" if st.session_state.bulk_mode == "Manual" else "secondary"):
                st.session_state.bulk_mode = "Manual"
                st.rerun()

        # Ambil nilai mode dari session state
        mode = st.session_state.bulk_mode

        # Pilihan Job hanya muncul jika mode Manual
        # Pilihan Job hanya muncul jika mode Manual
        selected_job = None
        if mode == "Manual":
            # st.markdown("<br>", unsafe_allow_html=True) # Spasi tipis
            
            # Container visual untuk Dropdown
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 5px solid #0B5334; margin-bottom: -20px;'>
                    <p style='margin:0; font-size: 12px; color: gray; font-weight: bold;'>🎯 TARGET POSISI</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Selectbox asli Streamlit
            selected_job = st.selectbox("", df['job_title'], key="job_manual")
            
        # File Uploader
        files = st.file_uploader(
            "Upload banyak file CV (PDF)",
            type="pdf",
            accept_multiple_files=True,
            key=st.session_state.uploader_key
        )

        # Tombol Proses
        process_btn = st.button("🚀 Proses Screening", use_container_width=True, type="primary")
        
        # Tombol Reset
        if st.button("🔄 Reset & Hapus File", use_container_width=True):
            st.session_state.uploader_key += 1
            st.rerun()

        # --- KARTU RINGKASAN ---
        if files:
            st.markdown(f"""
            <div class="sidebar-stats-card">
                <p style="color: #0B5334; font-weight: bold; margin-bottom: 5px;">Total CV diupload</p>
                <h2 style="color: #0B5334; margin-top: -20px;">{len(files)} <small style="font-size: 14px; color: gray;">File</small></h2>
            </div>
            """, unsafe_allow_html=True)

    # --- LOGIC PROSES ---
    # --- LOGIC PROSES ---
    if files and process_btn:
        results = []
        with st.spinner("⏳⏳⏳Menganalisis semua CV..."):
            for f in files:
                text = extract_text_from_pdf(f)
                cv_extracted = parse_cv_structured(text)
                df_sim = compute_similarity(cv_extracted, job_embeddings, model,translation_tokenizer, translation_model, df)
                
                # Cari Rekomendasi Terbaik AI (Peringkat 1 Global) untuk tampilan di UI
                best_ai_row = df_sim.sort_values(by="score", ascending=False).iloc[0]
                
                if mode == "Otomatis":
                    # Di mode Otomatis, yang diproses dan disimpan adalah hasil deteksi AI terbaik
                    job_to_save = best_ai_row['job_title']
                    score_val = best_ai_row['score']
                    skill_val = best_ai_row['skill']
                    exp_val = best_ai_row['exp']
                    edu_val = best_ai_row['edu']
                else:
                    # Di mode Manual, yang diproses dan disimpan HANYA job yang dipilih user
                    row_target = df_sim[df_sim['job_title'] == selected_job].iloc[0]
                    job_to_save = selected_job
                    score_val = row_target['score']
                    skill_val = row_target['skill']
                    exp_val = row_target['exp']
                    edu_val = row_target['edu']
                # --- SIMPAN KE DATABASE (leaderboardbulk.csv) ---
                # Hanya simpan data job_target (tanpa kolom best_ai) agar seragam dengan Otomatis
                # Data untuk kebutuhan visualisasi UI
                res_ui = {
                    "cv": f.name,
                    "job_target": job_to_save,
                    "best_ai": best_ai_row['job_title'],
                    "score": max(0.0, round(score_val * 100, 2)),
                    "skill": max(0.0, round(skill_val * 100, 2)),
                    "exp":   max(0.0, round(exp_val   * 100, 2)),
                    "edu":   max(0.0, round(edu_val   * 100, 2))
                }
                results.append(res_ui)

                # # --- SIMPAN KE DATABASE (leaderboardbulk.csv) ---
                # Hanya simpan data job_target (tanpa kolom best_ai) agar seragam dengan Otomatis
                # Data untuk kebutuhan visualisasi UI
                data_db = {
                    "cv": f.name,
                    "job": job_to_save,
                    "score": max(0.0, round(score_val * 100, 2)),
                    "skill": max(0.0, round(skill_val * 100, 2)),
                    "exp":   max(0.0, round(exp_val   * 100, 2)),
                    "edu":   max(0.0, round(edu_val   * 100, 2))
                }
                save_data("leaderboardbulk.csv", data_db, ["cv", "job","score", "skill", "exp", "edu"])
        
        df_res = pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
        if df_res.empty:
            st.warning("⚠️ Tidak ada CV yang berhasil diproses.")
            st.stop()
        df_res["rank"] = df_res.index + 1

        # --- HEADER & STATS CARDS ---
        st.markdown(f'<p class="main-title">📊 Hasil Bulk CV Screening ({mode} Mode)</p>', unsafe_allow_html=True)
        
        top_val = df_res.iloc[0]
        total_cv = len(df_res)

        # Logika pembagian kolom (5 untuk Manual, 4 untuk Otomatis)
        if mode == "Manual":
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid #FFD700;">
                <p style="color: gray; font-weight: bold; margin-bottom:5px;">🏆 CV TERBAIK</p>
                <h4 style="margin:0; color:#333;">{top_val['cv'][:15]}</h4>
                <p style="color: gray; font-size: 11px; ">Kandidat skor tertinggi</p>
            </div>""", unsafe_allow_html=True)

        with col_m2:
            top_score_color = get_score_color(top_val['score'])
            st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid {top_score_color};">
                <p style="color: gray; font-weight: bold; margin-bottom:5px;">🥇 SKOR TERTINGGI</p>
                <h1 style="color: {top_score_color}; margin:0px 0; font-size:35px;">{top_val['score']:.1f}%</h1>
                <span style="background: #FFF9E6; border-radius: 20px; font-size: 12px; border: 1px solid #FFE082; font-weight:bold; color:#856404; padding: 2px 10px;">RANK #1</span>
            </div>""", unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid #333;">
                <p style="color: gray; font-weight: bold; margin-bottom:5px;">📄 TOTAL CV</p>
                <h1 style="color: #333; margin:0px 0; font-size:35px;">{total_cv}</h1>
                <p style="color: gray; font-size: 12px; margin:0;">File PDF diupload</p>
            </div>""", unsafe_allow_html=True)

        if mode == "Manual":
            with col_m4:
                st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid #28A745;">
                    <p style="color: gray; font-weight: bold; margin-bottom:5px;">🎯 JOB TARGET</p>
                    <h4 style="margin:0; color:#333;">{top_val['job_target']}</h4>
                    <p style="color: gray; font-size: 11px; margin-top:10px;">Pilihan Anda</p>
                </div>""", unsafe_allow_html=True)
            with col_m5:
                st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid #1E90FF;">
                    <p style="color: gray; font-weight: bold; margin-bottom:5px;">💡 REKOMENDASI</p>
                    <h4 style="margin:0; color:#1E90FF;">{top_val['best_ai']}</h4>
                    <p style="color: gray; font-size: 11px; margin-top:10px;">Paling Direkomendasikan</p>
                </div>""", unsafe_allow_html=True)
        else:
            with col_m4:
                st.markdown(f"""<div class="custom-card" style="text-align: center; border-top: 5px solid #28A745;">
                    <p style="color: gray; font-weight: bold; margin-bottom:5px;">💼 JOB TERBAIK</p>
                    <h4 style="margin:0; color:#333;">{top_val['job_target']}</h4>
                    <p style="color: gray; font-size: 11px; margin-top:10px;">Hasil Deteksi AI</p>
                </div>""", unsafe_allow_html=True)


        # --- GRAFIK DISTRIBUSI ---
        # st.subheader("📈 Distribusi Skor Seluruh Kandidat")
        st.markdown("""
            <h3 style="
                color:#0B5334;
                font-size:22px;
                font-weight:700;
                margin-bottom:10px;
            ">
            📈 Distribusi Skor Seluruh Kandidat
            </h3>
            """, unsafe_allow_html=True)
            
        chart = alt.Chart(df_res).mark_bar(color='#28A745', cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X("cv:N", sort="-y", title="Nama File CV"),
            y=alt.Y("score:Q", title="Skor (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["cv", "job_target", "score"]
        ).properties(height=600)
        
        text = chart.mark_text(baseline='bottom', dy=-5, fontWeight='bold').encode(text='score:Q')
        st.altair_chart(chart + text, use_container_width=True)

        # --- TABEL HASIL GAYA HIJAU KALLA ---
        # st.subheader("📋 Tabel Penilaian Keseluruhan")
        st.markdown("""
            <h3 style="
                color:#0B5334;
                font-size:22px;
                font-weight:700;
                margin-bottom:10px;
            ">
            📋 Tabel Penilaian Keseluruhan
            </h3>
            """, unsafe_allow_html=True)
        
        df_table = df_res.copy()
        for col in ["score", "skill", "exp", "edu"]:
            df_table[col] = df_table[col].apply(lambda x: f"{x:.2f}%")

        def style_bulk(df):
            return df.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0B5334'), ('color', 'white'), ('text-align', 'center !important'), ('padding', '15px')]},
                {'selector': 'td', 'props': [('text-align', 'center !important'), ('padding', '12px')]}
            ]).set_properties(**{'background-color': 'white', 'color': '#333', 'border-color': '#f0f0f0', 'text-align': 'center !important'})

        # Menentukan kolom tabel berdasarkan mode
        if mode == "Manual":
            final_cols = ["rank", "cv", "job_target", "best_ai", "score", "skill", "exp", "edu"]
            col_names = {"rank": "Rank", "cv": "Nama CV", "job_target": "Job Pilihan", "best_ai": "Saran Job", "score": "Skor CV", "skill": "Kemampuan", "exp": "Pengalaman", "edu": "Pendidikan"}
        else:
            final_cols = ["rank", "cv", "job_target", "score", "skill", "exp", "edu"]
            col_names = {"rank": "Rank", "cv": "Nama CV", "job_target": "Job Terbaik", "score": "Skor CV", "skill": "Kemampuan", "exp": "Pengalaman", "edu": "Pendidikan"}

        st.table(style_bulk(df_table[final_cols].rename(columns=col_names)))

        # --- RANGKUMAN DETEKSI JOB ---
        if mode == "Otomatis":
            # st.subheader("📊 Sebaran Deteksi Posisi")
            st.markdown("""
            <h3 style="
                color:#0B5334;
                font-size:22px;
                font-weight:700;
                margin-bottom:10px;
            ">
            📊 Sebaran Deteksi Posisi
            </h3>
            """, unsafe_allow_html=True)
            job_counts = df_res['job_target'].value_counts()
            cols_job = st.columns(4)
            for idx, (j_name, count) in enumerate(job_counts.items()):
                with cols_job[idx % 4]:
                    st.markdown(f"""<div style="background:white; padding:15px; border-radius:10px; border:1px solid #eee; margin-bottom:10px; text-align:center;">
                        <p style="margin:0; font-size:12px; color:gray; font-weight:bold;">{j_name.upper()}</p>
                        <h3 style="margin:5px 0; color:#0B5334;">{count} <span style="font-size:12px; font-weight:normal;">CV</span></h3>
                    </div>""", unsafe_allow_html=True)

        st.success(f"✅ Berhasil memproses {len(df_res)} CV!")

    elif not files:
        st.info("👋 Silakan upload beberapa file CV di sidebar untuk memulai analisis Bulk.")