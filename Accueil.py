import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
from pathlib import Path
import base64

# ------------------------------------------
# 🔧 CONFIGURATION DE LA PAGE
# ------------------------------------------
st.set_page_config(
    page_title="Open Data Culinary Road Trip",
    layout="wide"
)

# ------------------------------------------
# 📦 CHARGEMENT DU CSS
# ------------------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------
# 🖼️ HELPER : IMG → BACKGROUND
# ------------------------------------------
def media_div(image_path: str, fallback_gradient: str) -> str:
    path = Path(image_path)
    if path.exists():
        img_data = base64.b64encode(path.read_bytes()).decode()
        return f"<div class='media' style=\"background-image: url('data:image/png;base64,{img_data}');\"></div>"
    else:
        return f"<div class='media' style=\"background-image: {fallback_gradient}\"></div>"



# ------------------------------------------
# 🏠 PAGE D’ACCUEIL – TITRE + INTRO
# ------------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:40px; margin-bottom:20px;">
    <h1 style="font-size: 3rem; font-weight: 700;"> Open Data Culinary Road Trip</h1>
    <p style="font-size:1.2rem; color:#555; margin-top:10px;">
       Explorez, découvrez et vivez une aventure gastronomique à travers l'Europe, guidée par les données Open Data.
    </p>
    <p style="font-size:1.1rem; color:#777;">
       Cartes interactives, itinéraires sur mesure, analyse des tendances culinaires et suggestions personnalisées.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# ✨ SECTION : Découvrir les fonctionnalités
# ------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <h2 style="font-size:2rem; font-weight:600;"> Découvrir les fonctionnalités</h2>
    <p style="color:#666; font-size:1.1rem;">
        Choisissez un module ci-dessous pour explorer les outils culinaires mis à votre disposition.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🟩 GRID DES CARTES
# ------------------------------------------
st.markdown('<div class="grid">', unsafe_allow_html=True)

# ---------------------- CARD 1 ----------------------
st.markdown(
    "<a href='/Maps' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/roadtrip.jpg",
        "radial-gradient(circle at 20% 30%, #d4e5dd 0%, #d4e5dd 25%, #f6f7f8 26%, #f6f7f8 100%)"
      )
    + """
      <div class="body">
        <h3>Carte interactive</h3>
        <p>Visualisez des milliers de restaurants européens sur une carte dynamique.
           Filtrez par cuisine, prix, note ou région pour planifier vos arrêts culinaires.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD 2 ----------------------
st.markdown(
    "<a href='./Roadtrip2' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/resto3.jpg",
        "radial-gradient(circle at 70% 20%, #ffd6c2 0%, #ffd6c2 18%, #f7e6e0 19%, #f7e6e0 100%)"
      )
    + """
      <div class="body">
        <h3>Road Trip Culinaire</h3>
        <p>Créez un itinéraire gourmand sur plusieurs jours.
           Sélectionnez des pays, des villes, des cuisines et obtenez un parcours optimisé.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD 3 ----------------------
st.markdown(
    "<a href='/Stats' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/resto2.jpg",
        "radial-gradient(circle at 30% 40%, #ffe7a0 0%, #ffe7a0 14%, #f3f4f6 15%, #f3f4f6 100%)"
      )
    + """
      <div class="body">
        <h3>Profil Gourmet</h3>
        <p>Indiquez vos goûts, votre budget et vos préférences.
           Recevez des recommandations de restaurants adaptés à votre identité culinaire.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD ' ----------------------
st.markdown(
    "<a href='/Top5' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/resto.jpg",
        "radial-gradient(circle at 30% 40%, #ffe7a0 0%, #ffe7a0 14%, #f3f4f6 15%, #f3f4f6 100%)"
      )
    + """
      <div class="body">
        <h3>Profil Gourmet</h3>
        <p>Indiquez vos goûts, votre budget et vos préférences.
           Recevez des recommandations de restaurants adaptés à votre identité culinaire.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# 🔚 Fin des containers
# ------------------------------------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
