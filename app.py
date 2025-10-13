import streamlit as st
import pandas as pd

st.set_page_config(page_title="🍽️ Culinary Road Trip", page_icon="🍴", layout="wide")

st.title("🍽️ Culinary Road Trip")
st.write("Bienvenue dans votre exploration culinaire en Europe à partir de données **Open Data** 🍷🇫🇷🇮🇹🇪🇸")
st.sidebar.header("🔎 Filtres")
country = st.sidebar.text_input("Pays")
city = st.sidebar.text_input("Ville")
df = pd.DataFrame(columns=["Nom", "Ville", "Pays", "Note", "Prix"])

if country or city:
    st.success(f"Recherche de restaurants pour {city or '...'}, {country or '...'}")
    st.dataframe(df)
else:
    st.info("Saisis un pays ou une ville pour commencer.")

st.markdown("---")
st.caption("Projet Open Data — MIASHS 2025")
