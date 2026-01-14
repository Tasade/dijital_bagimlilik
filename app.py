import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Tech Addiction • Sleep & Health Dashboard", layout="wide")

st.title("📱 Teknoloji Bağımlılığı • Uyku • Sağlık Analitiği")
st.caption("Bu panel: bağımlılık göstergeleri ile uyku/mental sağlık metrikleri arasındaki ilişkiyi görselleştirir.")

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

uploaded = st.sidebar.file_uploader("CSV yükle (mobile_addiction_data.csv)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # aynı klasördeyse otomatik okur
    try:
        df = load_data_from_path("mobile_addiction_data.csv")
    except Exception:
        st.warning("CSV bulunamadı. Soldan dosyayı yükleyebilirsin.")
        st.stop()

# -----------------------------
# Basic cleaning
# -----------------------------
# Saat/usage alanlarında bazen negatif değer olabiliyor → 0'a kırp (mantıksal temizlik)
hour_cols = [
    "Daily_Screen_Time_Hours",
    "Social_Media_Usage_Hours",
    "Gaming_Usage_Hours",
    "Streaming_Usage_Hours",
    "Messaging_Usage_Hours",
    "Work_Related_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Time_Spent_With_Family_Hours",
    "Online_Shopping_Hours",
]
for c in hour_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].clip(lower=0)

# Numerik bazı alanları garantiye al
numeric_cols = [
    "Phone_Unlocks_Per_Day",
    "Push_Notifications_Per_Day",
    "Mental_Health_Score",
    "Depression_Score",
    "Anxiety_Score",
    "Stress_Level",
    "Income_USD",
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Age", "Gender", "Self_Reported_Addiction_Level"])

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filtreler")

countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
sel_country = st.sidebar.multiselect("Ülke", countries, default=countries[:10] if len(countries) > 10 else countries)

genders = sorted(df["Gender"].dropna().unique())
sel_gender = st.sidebar.multiselect("Cinsiyet", genders, default=genders)

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Yaş Aralığı", min_age, max_age, (min_age, max_age))

add_levels = ["Low", "Moderate", "High", "Severe"]
available_levels = [x for x in add_levels if x in set(df["Self_Reported_Addiction_Level"].unique())]
sel_add = st.sidebar.multiselect("Bağımlılık Seviyesi", available_levels, default=available_levels)

filtered = df.copy()
if sel_country and "Country" in filtered.columns:
    filtered = filtered[filtered["Country"].isin(sel_country)]
filtered = filtered[filtered["Gender"].isin(sel_gender)]
filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]
filtered = filtered[filtered["Self_Reported_Addiction_Level"].isin(sel_add)]

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kayıt sayısı", f"{len(filtered):,}")
c2.metric("Ortalama Uyku (saat)", f"{filtered['Sleep_Hours'].mean():.2f}" if "Sleep_Hours" in filtered.columns else "—")
c3.metric("Ortalama Mental Sağlık", f"{filtered['Mental_Health_Score'].mean():.2f}" if "Mental_Health_Score" in filtered.columns else "—")
c4.metric("Ortalama Sosyal Medya (saat)", f"{filtered['Social_Media_Usage_Hours'].mean():.2f}" if "Social_Media_Usage_Hours" in filtered.columns else "—")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🛌 Uyku", "🧠 Sağlık", "📲 Sosyal Medya", "🔗 Korelasyon"])

# ---------
# TAB 1: Sleep
# ---------
with tab1:
    st.subheader("Bağımlılık Seviyesi → Uyku")
    if "Sleep_Hours" in filtered.columns:
        grp = filtered.groupby("Self_Reported_Addiction_Level")["Sleep_Hours"].mean().reindex(add_levels).dropna()

        fig, ax = plt.subplots()
        grp.plot(kind="bar", ax=ax)
        ax.set_xlabel("Bağımlılık Seviyesi")
        ax.set_ylabel("Ortalama Uyku (saat)")
        st.pyplot(fig)

        st.markdown("**Sosyal medya saatine göre uyku** (çeyreklik gruplar)")
        # Social media quartiles → sleep
        if "Social_Media_Usage_Hours" in filtered.columns:
            q = pd.qcut(filtered["Social_Media_Usage_Hours"], 4, duplicates="drop")
            tmp = filtered.assign(SM_Quartile=q).groupby("SM_Quartile")["Sleep_Hours"].mean()

            fig, ax = plt.subplots()
            tmp.plot(kind="bar", ax=ax)
            ax.set_xlabel("Sosyal Medya Kullanımı (Çeyreklik)")
            ax.set_ylabel("Ortalama Uyku (saat)")
            st.pyplot(fig)

    st.subheader("Night Mode Açık/Kapalı → Uyku farkı")
    if "Has_Night_Mode_On" in filtered.columns and "Sleep_Hours" in filtered.columns:
        nm = filtered.groupby("Has_Night_Mode_On")["Sleep_Hours"].mean()

        fig, ax = plt.subplots()
        nm.plot(kind="bar", ax=ax)
        ax.set_xlabel("Night Mode")
        ax.set_ylabel("Ortalama Uyku (saat)")
        st.pyplot(fig)

# ---------
# TAB 2: Health
# ---------
with tab2:
    st.subheader("Bağımlılık Seviyesi → Mental Sağlık / Depresyon / Anksiyete / Stres")
    metrics = [c for c in ["Mental_Health_Score", "Depression_Score", "Anxiety_Score", "Stress_Level"] if c in filtered.columns]

    if metrics:
        left, right = st.columns(2)
        with left:
            metric_sel = st.selectbox("Metriği seç", metrics, index=0)
            grp = filtered.groupby("Self_Reported_Addiction_Level")[metric_sel].mean().reindex(add_levels).dropna()

            fig, ax = plt.subplots()
            grp.plot(kind="bar", ax=ax)
            ax.set_xlabel("Bağımlılık Seviyesi")
            ax.set_ylabel(f"Ortalama {metric_sel}")
            st.pyplot(fig)

        with right:
            st.markdown("**Ekran süresi ile seçili sağlık metriği ilişkisi**")
            xcol = "Daily_Screen_Time_Hours" if "Daily_Screen_Time_Hours" in filtered.columns else None
            ycol = metric_sel
            if xcol:
                fig, ax = plt.subplots()
                ax.scatter(filtered[xcol], filtered[ycol], s=6)
                ax.set_xlabel("Günlük Ekran Süresi (saat)")
                ax.set_ylabel(ycol)

                # basit trend çizgisi (linear fit)
                x = filtered[xcol].dropna().values
                y = filtered.loc[filtered[xcol].notna(), ycol].values
                if len(x) > 5:
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xs, m*xs + b)
                st.pyplot(fig)

    st.subheader("Telefon açma sayısı → Stres")
    if "Phone_Unlocks_Per_Day" in filtered.columns and "Stress_Level" in filtered.columns:
        fig, ax = plt.subplots()
        ax.scatter(filtered["Phone_Unlocks_Per_Day"], filtered["Stress_Level"], s=6)
        ax.set_xlabel("Günlük Telefon Açma Sayısı")
        ax.set_ylabel("Stres Seviyesi")
        st.pyplot(fig)

# ---------
# TAB 3: Social Media
# ---------
with tab3:
    st.subheader("Sosyal medya kullanımı → Sağlık (seç ve gör)")
    y_metrics = [c for c in ["Sleep_Hours", "Mental_Health_Score", "Depression_Score", "Anxiety_Score", "Stress_Level"] if c in filtered.columns]
    if "Social_Media_Usage_Hours" in filtered.columns and y_metrics:
        ysel = st.selectbox("Y ekseni metriği", y_metrics, index=0)

        fig, ax = plt.subplots()
        ax.scatter(filtered["Social_Media_Usage_Hours"], filtered[ysel], s=6)
        ax.set_xlabel("Sosyal Medya Kullanımı (saat)")
        ax.set_ylabel(ysel)

        # trend
        x = filtered["Social_Media_Usage_Hours"].dropna().values
        y = filtered.loc[filtered["Social_Media_Usage_Hours"].notna(), ysel].values
        if len(x) > 5:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m*xs + b)
        st.pyplot(fig)

    st.subheader("Bağımlılık seviyesi → Sosyal medya saatleri")
    if "Social_Media_Usage_Hours" in filtered.columns:
        grp = filtered.groupby("Self_Reported_Addiction_Level")["Social_Media_Usage_Hours"].mean().reindex(add_levels).dropna()
        fig, ax = plt.subplots()
        grp.plot(kind="bar", ax=ax)
        ax.set_xlabel("Bağımlılık Seviyesi")
        ax.set_ylabel("Ortalama Sosyal Medya (saat)")
        st.pyplot(fig)

# ---------
# TAB 4: Correlation
# ---------
with tab4:
    st.subheader("Sayısal değişken korelasyonları (hızlı teşhis)")
    numeric = filtered.select_dtypes(include=[np.number]).copy()
    # Çok sütun çıkarsa, ana alanları seçerek sadeleştir
    main_cols = [c for c in [
        "Daily_Screen_Time_Hours", "Social_Media_Usage_Hours", "Phone_Unlocks_Per_Day",
        "Sleep_Hours", "Mental_Health_Score", "Depression_Score", "Anxiety_Score",
        "Stress_Level", "Physical_Activity_Hours", "Time_Spent_With_Family_Hours",
        "Push_Notifications_Per_Day"
    ] if c in numeric.columns]

    if len(main_cols) >= 2:
        corr = numeric[main_cols].corr()
        st.dataframe(corr.style.format("{:.2f}"))
        st.caption("İpucu: |corr| yüksekse ilişki güçlü olabilir; ama nedensellik kanıtı değildir.")
    else:
        st.info("Korelasyon için yeterli sayısal sütun bulunamadı.")

st.divider()
st.subheader("Veri Önizleme")
st.dataframe(filtered.head(200))
