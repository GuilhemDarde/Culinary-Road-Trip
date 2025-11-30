import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test HuggingFace CSV", layout="wide")

st.title("🧪 Test — Chargement du CSV depuis Hugging Face Datasets")

@st.cache_data(ttl=3600)
def load_data():
    
    df = pd.read_csv("hf://datasets/Amoham16/aya-culinary-trip/tripadvisor_european_restaurants.csv")
    return df 

st.write("📥 Tentative de chargement du CSV...")

try:
    df = load_data()
    st.success("✅ CSV chargé avec succès depuis Hugging Face !")

    st.markdown("### Aperçu des données")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Informations générales")
    st.write(f"• Nombre de lignes : **{len(df)}**")
    st.write(f"• Colonnes : {list(df.columns)}")

except Exception as e:
    st.error("❌ Erreur lors du chargement du CSV.")
    st.exception(e)
