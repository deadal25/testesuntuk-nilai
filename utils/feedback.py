from utils.scoring import interpret_score

def generate_feedback(cv_parts, best_job, score, skill, exp, edu):
    s_pct, e_pct, d_pct = skill * 100, exp * 100, edu * 100
    feedback = f"""
    <div style='line-height:1.30; margin:0; padding:0; text-align:justify;'>
    CV Anda menunjukkan profil akademik sebagai lulusan {cv_parts['degree'].upper()} {str(cv_parts['major']).title()}. 
    Berdasarkan hasil analisis terhadap posisi <b>{best_job}</b>, sistem memberikan skor kesesuaian sebesar <b>{float(score):.2f}%</b> dengan kategori <b>{interpret_score(float(score))}</b>. Hasil tersebut menunjukkan tingkat relevansi antara profil kandidat dengan kebutuhan posisi yang dianalisis berdasarkan kemampuan, pengalaman, dan latar belakang pendidikan yang tercantum pada CV.
    """
    kelebihan = []
    if s_pct >= 70:
        kelebihan.append(f"penguasaan kemampuan teknis yang sangat baik ({float(s_pct):.2f}%)")
    if e_pct >= 70:
        kelebihan.append(f"pengalaman kerja yang relevan dengan posisi ({float(e_pct):.2f}%)")
    if d_pct >= 70:
        kelebihan.append(f"latar belakang pendidikan yang cukup linier ({float(d_pct):.2f}%)")
        
    if kelebihan:
        feedback += f"<br><b>Kekuatan Utama:</b> Profil Anda memiliki keunggulan pada " + ", serta ".join(kelebihan) + ". Hal ini menunjukkan bahwa CV Anda telah memiliki kompetensi yang cukup baik dan sesuai dengan kebutuhan pekerjaan."
    
    feedback += "<br><b>Analisis Pengembangan:</b> "
    if s_pct < 60:
        feedback += f"Aspek kemampuan teknis masih berada pada angka {float(s_pct):.2f}%. Disarankan untuk meningkatkan skill yang lebih relevan dengan posisi yang dituju. "
    else:
        feedback += "Kemampuan teknis Anda sudah cukup kompetitif untuk posisi ini. Namun, pengembangan skill terbaru tetap diperlukan agar sesuai dengan kebutuhan industri. "
        
    if e_pct < 60:
        feedback += f"Pengalaman kerja memperoleh skor {float(e_pct):.2f}% yang masih belum optimal. Penambahan proyek atau pengalaman profesional dapat membantu memperkuat profil Anda. "
    if d_pct < 60:
        feedback += f"Kesesuaian latar belakang pendidikan berada pada angka {float(d_pct):.2f}%. Pelatihan tambahan dan sertifikasi dapat membantu meningkatkan relevansi akademik Anda. "
        
    feedback += f"""
    <br><b>Saran Strategis:</b> Profil Anda sudah cukup baik untuk posisi ini, namun CV masih dapat diperkuat dengan menambahkan deskripsi pengalaman yang lebih spesifik dan terukur. Cantumkan pencapaian, tanggung jawab utama, atau hasil kerja yang pernah dicapai agar recruiter dapat melihat kemampuan dan kontribusi Anda secara lebih jelas.</div>
    """
    return feedback