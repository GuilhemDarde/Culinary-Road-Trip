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
    <h1 style="font-size: 3rem; font-weight: 700;">Open Data Culinary Road Trip</h1>
    <p style="font-size:1.2rem; color:#555; margin-top:10px;">
       Une application de data visualisation construite à partir de données ouvertes TripAdvisor
       sur les restaurants en Europe.
    </p>
    <p style="font-size:1.1rem; color:#777;">
    <br>
       Explorez la géographie des restaurants, analysez les tendances culinaires,
       planifiez un road trip gourmand et obtenez des recommandations adaptées à votre profil.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# ✨ SECTION : Découvrir les fonctionnalités
# ------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <h2 style="font-size:2rem; font-weight:600;">Découvrir les fonctionnalités</h2>
    <p style="color:#666; font-size:1.1rem;">
    <br>
        Naviguez entre les modules pour explorer les restaurants européens sous différents angles :
        cartographie, itinéraires, statistiques et suggestions personnalisées.
    <br>
    <br>
    <br>
    <br>
    <br>
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------
# 🟩 GRID DES CARTES
# ------------------------------------------
st.markdown('<div class="grid">', unsafe_allow_html=True)

# ---------------------- CARD 1 : CARTE ----------------------
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
        <p>Explorez la répartition des restaurants en Europe sur une carte dynamique.
           Filtrez par pays, ville, type de cuisine, niveau de prix et note moyenne.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD 2 : ROADTRIP ----------------------
st.markdown(
    "<a href='./Roadtrip2' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/resto.jpg",
        "radial-gradient(circle at 70% 20%, #ffd6c2 0%, #ffd6c2 18%, #f7e6e0 19%, #f7e6e0 100%)"
      )
    + """
      <div class="body">
        <h3>Road Trip Culinaire</h3>
        <p>Construisez un itinéraire gourmand sur plusieurs jours.
           Choisissez vos pays, vos villes et vos cuisines préférées, puis laissez l’algorithme proposer
           une sélection de restaurants pour chaque étape.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD 3 : STATS ----------------------
st.markdown(
    "<a href='/Stats' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/map2.jpg",
        "radial-gradient(circle at 30% 40%, #ffe7a0 0%, #ffe7a0 14%, #f3f4f6 15%, #f3f4f6 100%)"
      )
    + """
      <div class="body">
        <h3>Statistiques & tendances</h3>
        <p>Analysez les pays les plus représentés, les cuisines les mieux notées
           et la distribution des avis. Un module pensé pour explorer les tendances
           culinaires à l’échelle européenne.</p>
      </div>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)

# ---------------------- CARD 4 : TOP 5 / PROFIL ----------------------
st.markdown(
    "<a href='/Top5' target='_self' class='card-link'>"
    "<div class='card clickable-card'>"
    + media_div(
        "images/resto2.jpg",
        "radial-gradient(circle at 30% 40%, #ffe7a0 0%, #ffe7a0 14%, #f3f4f6 15%, #f3f4f6 100%)"
      )
    + """
      <div class="body">
        <h3>Top 5 personnalisé</h3>
        <p>Renseignez vos envies (ambiance, budget, type de cuisine).
           Obtenez une sélection de restaurants recommandés qui correspondent à votre profil gourmand.</p>
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
