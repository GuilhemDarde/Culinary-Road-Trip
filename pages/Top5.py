import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# Pour la carte Leaflet
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Top Restaurants", layout="wide")

# Charger le CSS externe
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================
# 🔹 CHARGEMENT DES DONNÉES
# ==============================
@st.cache_data(ttl=3600, max_entries=1)
def load_data():
    usecols = [
        "restaurant_name", "country", "region", "city",
        "latitude", "longitude",
        "avg_rating", "total_reviews_count",
        "price_level", "cuisines"
    ]
      # Téléchargement depuis Hugging Face Hub
    local_path = hf_hub_download(
        repo_id="Amoham16/dataset-resto-10k",
        repo_type="dataset",
        filename="tripadvisor_clean.csv",
    )
    
    df = pd.read_csv(local_path, usecols=usecols)

    # Nettoyage
    df = df.dropna(subset=["avg_rating"])
    df["price_level"] = df["price_level"].fillna("Inconnu")
    df["region"] = df["region"].fillna("Inconnue")
    df["city"] = df["city"].fillna("Inconnue")
    df["country"] = df["country"].fillna("Inconnu")
    df["cuisines"] = df["cuisines"].fillna("Inconnue")

    # Type de cuisine principal (premier de la liste)
    df["cuisines_clean"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip())

    return df


df = load_data()

# ==============================
# 🔹 UI & FILTRES
# ==============================
st.title("Trending Restaurants")

st.write("Filtre les restaurants selon tes préférences, puis affiche un Top N.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    cuisine = st.selectbox(
        "🍽 Type de cuisine",
        ["Tous"] + sorted(df["cuisines_clean"].unique().tolist())
    )

with col2:
    country = st.selectbox(
        "Pays",
        ["France"] + sorted(df["country"].unique().tolist())
    )

# Liste des villes dépendante du pays
if country != "Tous pays":
    cities_for_country = (
        df[df["country"] == country]["city"]
        .dropna()
        .unique()
        .tolist()
    )
else:
    cities_for_country = df["city"].dropna().unique().tolist()

cities_for_country = sorted(cities_for_country)

with col3:
    city = st.selectbox(
        "Ville",
        ["Toutes villes"] + cities_for_country
    )

with col4:
    top_n = st.select_slider(
        "Top N",
        options=[3, 5, 10],
        value=5
    )

# Filtre additionnel pour la note (en dessous)
min_rating = st.slider(
    "⭐ Note minimum",
    min_value=0.0, max_value=5.0,
    value=4.0, step=0.1
)


# ==============================
# 🔹 APPLICATION DES FILTRES
# ==============================
df_filtered = df.copy()

if cuisine != "Tous":
    df_filtered = df_filtered[df_filtered["cuisines_clean"] == cuisine]

if country != "Tous pays":
    df_filtered = df_filtered[df_filtered["country"] == country]

if city != "Toutes villes":
    df_filtered = df_filtered[df_filtered["city"] == city]

df_filtered = df_filtered[df_filtered["avg_rating"] >= min_rating]

# ==============================
# 🔹 TOP N
# ==============================
df_top = df_filtered.sort_values(
    by=["avg_rating", "total_reviews_count"],
    ascending=[False, False]
).head(top_n)

st.subheader(f"✨ Top {len(df_top)} restaurants correspondant aux critères")

if df_top.empty:
    st.warning("Aucun restaurant ne correspond à ces filtres.")
else:
    # Affichage des cartes résultat
    for _, row in df_top.iterrows():
        st.markdown(
            f"""
            <div class="result-card">
                <h3>{row['restaurant_name']}</h3>
                <p>{row['cuisines_clean']} • {row['city']} • {row['country']} • {row['price_level']}</p>
                <p><strong>⭐ {row['avg_rating']:.1f}</strong> — {int(row['total_reviews_count'])} avis</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==============================
    # 🔹 MINI CARTE LEAFLET
    # ==============================
    st.subheader("Localisation sur la carte")

    # On ne garde que ceux qui ont des coordonnées
    df_map = df_top.dropna(subset=["latitude", "longitude"])

    if df_map.empty:
        st.info("Pas de coordonnées disponibles pour afficher la carte.")
    else:
        # Centre de la carte = moyenne des lat/lon
        center_lat = df_map["latitude"].mean()
        center_lon = df_map["longitude"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

        for _, row in df_map.iterrows():
            popup_html = f"""
            <b>{row['restaurant_name']}</b><br>
            {row['cuisines_clean']}<br>
            {row['city']} – {row['country']}<br>
            ⭐ {row['avg_rating']:.1f} ({int(row['total_reviews_count'])} avis)
            """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup_html
            ).add_to(m)

        st_folium(m, width=900, height=400)
