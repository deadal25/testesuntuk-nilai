import re
from utils.pdf_extractor import clean_text

DEGREE_TRANSLATE_MAP = {
    "s1": "bachelor's degree",
    "s2": "master's degree",
    "d4": "applied bachelor's degree",
    "d3": "diploma",
    "sma": "senior high school",
    "smk": "vocational school",
    "smp": "junior high school",
    "sederajat": "equivalent",
}

def extract_section(text, start_keywords, end_keywords):
    """Memotong bagian teks tertentu berdasarkan jangkar kata kunci awal dan akhir."""
    text_lower = text.lower()
    start_pos = -1
    matched_keyword = ""
    for kw in start_keywords:
        pos = text_lower.find(kw.lower())
        if pos != -1:
            start_pos = pos
            matched_keyword = kw
            break
    if start_pos == -1:
        return ""
    start_pos += len(matched_keyword)

    end_pos = len(text)
    for end_kw in end_keywords:
        pos = text_lower.find(end_kw.lower(), start_pos)
        if pos != -1 and pos < end_pos:
            end_pos = pos
    return text[start_pos:end_pos].strip(" :,-\n\t")

# --- KODE BARU / PENYESUAIAN SESUAI COLAB ---
def extract_education_text(text):
    """Mengisolasi bagian Riwayat Pendidikan."""
    start_keywords = ["riwayat pendidikan"]
    end_keywords = [
        "pengalaman pekerjaan", "pengalaman organisasi",
        "penguasaan bahasa", "pekerjaan yang disukai",
        "tpa & toefl", "tpa dan toefl",
    ]
    return extract_section(text, start_keywords, end_keywords)

def extract_experience(text):
    """Mengisolasi bagian Pengalaman Kerja dan Organisasi."""
    pekerjaan = extract_section(text, ["pengalaman pekerjaan"], ["pengalaman organisasi", "no. identitas", "email", "kode pos", "penguasaan bahasa", "tpa & toefl", "pekerjaan yang disukai"])
    organisasi = extract_section(text, ["pengalaman organisasi"], ["pengalaman pekerjaan", "penguasaan bahasa", "tpa & toefl", "pekerjaan yang disukai"])
    if pekerjaan and organisasi:
        return f"{pekerjaan}, {organisasi}"
    return pekerjaan if pekerjaan else organisasi

def extract_skill(text):
    """Mengisolasi bagian Kemampuan Bahasa dan Pekerjaan yang Disukai."""
    skill = extract_section(text, ["penguasaan bahasa"], ["pekerjaan yang disukai"])
    pekerjaan_disukai = extract_section(text, ["pekerjaan yang disukai"], ["pengalaman organisasi", "no. identitas", "email", "kode pos"])
    if skill and pekerjaan_disukai:
        return f"{skill}, {pekerjaan_disukai}"
    return skill if skill else pekerjaan_disukai

def remove_section_labels(text):
    """Menghapus label teks struktural agar tidak mengotori representasi vektor embeddings."""
    labels = ["pengalaman pekerjaan", "pengalaman organisasi", "penguasaan bahasa", "pekerjaan yang disukai", "riwayat pendidikan"]
    for label in labels:
        text = re.sub(label, "", text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def parse_cv_structured(text):
    """Pipeline penataan komponen teks CV mentah ke dalam bentuk Dictionary."""
    # Sekarang menggunakan extract_education_text terpisah seperti di Colab
    education_text = extract_education_text(text)
    edu_lower = education_text.lower()

    degree = ""
    if re.search(r"\bs1\b", edu_lower): degree = "s1"
    elif re.search(r"\bd4\b", edu_lower) or re.search(r"\bd-iv\b", edu_lower): degree = "d4"
    elif re.search(r"\bs2\b", edu_lower): degree = "s2"
    elif re.search(r"\bd3\b", edu_lower) or re.search(r"\bd-iii\b", edu_lower): degree = "d3"
    elif re.search(r"\bsma\b", edu_lower): degree = "sma"
    elif re.search(r"\bsmk\b", edu_lower): degree = "smk"
    elif re.search(r"\bsmp\b", edu_lower): degree = "smp"

    major_keywords = [
        "sistem informasi", "teknik informatika", "ilmu komputer", "informatika", "manajemen sumber daya manusia aparatur", "statistika", "manajemen keuangan dan erbankan", "ilmu peternakan", "kesejahteraan sosial", "mekatronika dan kecerdasan buatan",
        "manajemen keuangan", "pendidikan teknik sipil", "teknik sipil", "teknik industri", "kesehatan masyarakat", "sosiologi", "teknik perencanaan wilayah dan kota", "management", "administrasi publik", "sastra arab",
        "teknik mesin", "budidaya pertanian", "teknik elektro", "akuntansi", "hukum", "administrasi negara", "ilmu komunikasi", "ilmu hubungan internasional", "ekonomi pembangunan",
        "psikologi", "ekonomi", "manajemen keuangan dan perbankan", "manajemen", "ilmu hukum", "bisnis", "ipa"
    ]

    major = ""
    for m in major_keywords:
        if m in edu_lower:
            major = m
            break

    # Menggunakan fungsi modular baru yang diekstrak terpisah
    pengalaman_raw = remove_section_labels(extract_experience(text))
    kemampuan_raw  = remove_section_labels(extract_skill(text))

    pengalaman = clean_text(pengalaman_raw)
    kemampuan  = clean_text(kemampuan_raw)

    if kemampuan.strip() == "":
        kemampuan = pengalaman

    edu = f"{degree}, {major}".strip(", ")
    return {"degree": degree, "major": major, "edu": edu, "exp": pengalaman, "skill": kemampuan}