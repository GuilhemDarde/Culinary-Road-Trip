import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Stats & Visualisations", layout="wide")

# Charger le CSS externe
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==========================
# 🔹 Chargement des données (rapide + cache)
# ==========================
@st.cache_data(ttl=3600)
def load_data():
    usecols = [
        "restaurant_name", "country", "city",
        "latitude", "longitude",
        "avg_rating", "total_reviews_count",
        "cuisines"
    ]

    df = pd.read_csv("tripadvisor_european_restaurants.csv", usecols=usecols)

    df = df.dropna(subset=["latitude", "longitude"])
    df["cuisines"] = df["cuisines"].fillna("Inconnue")
    df["cuisines_clean"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip())
    df["city"] = df["city"].fillna("Inconnue")
    df["country"] = df["country"].fillna("Inconnu")

    return df

df = load_data()


# ==========================
# 🔹 Filtres généraux
# ==========================
st.title("🏙️ Carte 3D — Visualisation des restaurants")

col1, col2, col3 = st.columns(3)

with col1:
    cuisine = st.selectbox(
        "🍽 Type de cuisine",
        ["Toutes"] + sorted(df["cuisines_clean"].unique())
    )

with col2:
    country = st.selectbox(
        "🌍 Pays",
        ["Tous"] + sorted(df["country"].unique())
    )

with col3:
    min_rating = st.slider(
        "⭐ Note minimum", 0.0, 5.0, 4.0, step=0.1
    )

# Appliquer filtres
df_filtered = df.copy()

if cuisine != "Toutes":
    df_filtered = df_filtered[df_filtered["cuisines_clean"] == cuisine]

if country != "Tous":
    df_filtered = df_filtered[df_filtered["country"] == country]

df_filtered = df_filtered[df_filtered["avg_rating"] >= min_rating]


# ==========================
# 🔹 Choix du mode d'affichage
# ==========================
st.subheader("🎛️ Mode d'affichage (hauteur des colonnes)")

mode = st.selectbox(
    "Afficher la hauteur en fonction de :",
    [
        "Nombre d'avis",
        "Note moyenne",
        "Popularité (note × avis)",
        "Uniforme"
    ]
)

# Calcul de la colonne d’altitude
if mode == "Nombre d'avis":
    df_filtered["height"] = df_filtered["total_reviews_count"]

elif mode == "Note moyenne":
    df_filtered["height"] = df_filtered["avg_rating"] * 20  # scaling pour visibilité

elif mode == "Popularité (note × avis)":
    df_filtered["height"] = df_filtered["avg_rating"] * df_filtered["total_reviews_count"] / 5

else:  # Uniforme
    df_filtered["height"] = 50  # hauteur fixe

# Normalisation si valeurs très grandes
if df_filtered["height"].max() > 5000:
    df_filtered["height"] = df_filtered["height"] / 50


# ==========================
# 🔹 Carte 3D unique
# ==========================
st.subheader("🗺️ Carte 3D")

if df_filtered.empty:
    st.warning("Aucun restaurant trouvé avec ces filtres.")
    st.stop()

view = pdk.ViewState(
    latitude=df_filtered["latitude"].mean(),
    longitude=df_filtered["longitude"].mean(),
    zoom=4,
    pitch=55,
)

layer_3d = pdk.Layer(
    "ColumnLayer",
    data=df_filtered,
    get_position=["longitude", "latitude"],
    get_elevation="total_reviews_count",
    elevation_scale=15,
    radius=12000,
    get_color=[60, 120, 255, 180],
    pickable=True,
    auto_highlight=True,
)

view_3d = pdk.ViewState(
    latitude=df_filtered["latitude"].mean(),
    longitude=df_filtered["longitude"].mean(),
    zoom=4,
    pitch=55,   # inclinaison pour le côté 3D
)

deck_3d = pdk.Deck(
    initial_view_state=view,
    layers=[layer_3d],
    tooltip={"text": "{restaurant_name}\n⭐ {avg_rating}\nAvis: {total_reviews_count}"}
)

st.pydeck_chart(deck_3d)

# ==========================
# 🔹 ANALYSES GRAPHIQUES
# ==========================
st.subheader("📊 Analyses complémentaires")

# --- AGRÉGATIONS ---
df_country = (
    df_filtered.groupby("country")
    .agg(
        avg_rating_mean=("avg_rating", "mean"),
        total_reviews_sum=("total_reviews_count", "sum"),
        count_resto=("restaurant_name", "count")
    )
    .sort_values(by="avg_rating_mean", ascending=False)
)

df_cuisine = (
    df_filtered.groupby("cuisines_clean")
    .agg(
        avg_rating_mean=("avg_rating", "mean"),
        total_reviews_sum=("total_reviews_count", "sum"),
        count_resto=("restaurant_name", "count")
    )
    .sort_values(by="avg_rating_mean", ascending=False)
)

# ==========================
# 🔹 GRAPH 1 – Top pays par note moyenne
# ==========================
st.markdown("### 🇪🇺 Top pays par note moyenne")
st.bar_chart(df_country["avg_rating_mean"].head(10))

# ==========================
# 🔹 GRAPH 2 – Pays avec le plus d’avis
# ==========================
st.markdown("### 🗳️ Pays avec le plus d’avis")
st.bar_chart(df_country["total_reviews_sum"].head(10))

# ==========================
# 🔹 GRAPH 3 – Top cuisines par note
# ==========================
st.markdown("### 🍽️ Top types de cuisine par note moyenne")
st.bar_chart(df_cuisine["avg_rating_mean"].head(10))

# ==========================
# 🔹 GRAPH 4 – Distribution globale des notes
# ==========================
st.markdown("### ⭐ Distribution des notes")
st.histogram(df_filtered["avg_rating"])

