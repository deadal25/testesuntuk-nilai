import fitz  # PyMuPDF
import re

def extract_text_from_pdf(file):
    """Mengekstrak teks mentah secara linier dari file stream PDF."""
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            page_text = page.get_text()
            if page_text:
                text += page_text + " "
        pdf.close()
        text = re.sub(r'[\r\n]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print("Error reading PDF:", e)
    return text

def clean_text(text):
    """Membersihkan format teks dari noise penulisan tanggal/periode waktu."""
    text = str(text).lower()
    bulan_pattern = r'(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)'
    tahun_pattern = r'\b(200\d|201\d|202[0-6])\b'
    periode_pattern = f'{bulan_pattern}\s+{tahun_pattern}\s*(selesai|-|sampai)?\s*({bulan_pattern}\s+{tahun_pattern}|sekarang)?'
    
    text = re.sub(f'masuk\s*:\s*{periode_pattern}\s*(?=deskripsi\s*:\s*)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(f'mulai\s*:\s*{periode_pattern}\s*(?=tanggung\s*jawab\s*:\s*)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'deskripsi\s*:\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'tanggung\s*jawab\s*:\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(masuk|mulai|deskripsi|tanggung\s+jawab|selesai|sekarang)\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(f'\\b{bulan_pattern}\\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(f'{tahun_pattern}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Karakter pengganggu
    text = text.replace(';', ',').replace(':', '').replace('(', '').replace(')', '').replace('|', '').replace('-', '').replace('_', '').replace('--', '')
    return re.sub(r'\s+', ' ', text).strip()