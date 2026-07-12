def compute_weighted(df_sim):
    df_sim['weighted'] = (
        df_sim['skill'] * 0.5 +
        df_sim['exp'] * 0.4 +
        df_sim['edu'] * 0.1
    )
    return df_sim

def interpret_score(score):
    if score >= 75:
        return "Sangat Tinggi"
    elif score >= 60:
        return "Tinggi"
    elif score >= 40:
        return "Cukup"
    else:
        return "Rendah"