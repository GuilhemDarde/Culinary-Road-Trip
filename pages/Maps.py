import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import hf_hub_download

# ==============================
# 🔹 CONFIG
# ==============================
st.set_page_config(
    page_title="Carte interactive - Road Trip Culinaire",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_data():
    # Colonnes à charger
    usecols = [
        "restaurant_name", "country", "region", "city",
        "latitude", "longitude", "avg_rating", "total_reviews_count",
        "price_level", "cuisines"
    ]
    # Téléchargement depuis Hugging Face Hub
    local_path = hf_hub_download(
        repo_id="Amoham16/dataset-resto-10k",
        repo_type="dataset",
        filename="tripadvisor_clean.csv",
    )

    # Chargement depuis Hugging Face
    df = pd.read_csv(local_path, usecols=usecols)

    # Nettoyage et typage
    df = df.dropna(subset=["latitude", "longitude", "avg_rating"])
    df["price_level"] = df["price_level"].fillna("Inconnu")
    df["region"] = df["region"].fillna("Inconnue")
    df["country"] = df["country"].fillna("Inconnu")
    df["cuisines"] = df["cuisines"].fillna("Inconnue")

    # Colonnes catégorielles pour accélérer les filtres
    for col in ["country", "region", "city", "price_level"]:
        df[col] = df[col].astype("category")

    # Colonne simplifiée de cuisine
    df["cuisines_clean"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip())

    # Valeurs uniques pré-calculées pour les filtres
    country_list = sorted(df["country"].unique().tolist())
    cuisine_list = sorted(df["cuisines_clean"].unique().tolist())
    price_list = sorted(df["price_level"].unique().tolist())

    return df, country_list, cuisine_list, price_list


# ==============================
# 🔹 CHARGEMENT AVEC SPINNER
# ==============================
with st.spinner("Chargement des données... 🍽️"):
    df, country_list, cuisine_list, price_list = load_data()

st.success("Données prêtes à être explorées !")

# ==============================
# 🎛️ BARRE LATÉRALE DE FILTRES
# ==============================
st.sidebar.header("Filtres")



selected_countries = st.sidebar.multiselect(
    "Pays",
    country_list,
    default=["France"]  
)

# --- Filtre Région dépendant ---
if selected_countries:
    possible_regions = sorted(
        df[df["country"].isin(selected_countries)]["region"].unique().tolist()
    )
else:
    possible_regions = sorted(df["region"].unique().tolist())


selected_regions = st.sidebar.multiselect(
    "Région",
    options=possible_regions,
    default=[]
)

# --- Cuisine ---
selected_cuisines = st.sidebar.multiselect(
    "Cuisine",
    cuisine_list,
    default=[]  # pas de cuisine imposée au départ
)

# --- Prix (tous sélectionnés par défaut) ---
selected_prices = st.sidebar.multiselect(
    "Prix",
    price_list,
    default=price_list
)

# --- Note ---
min_rating = st.sidebar.slider(
    "Note minimale",
    0.0, 5.0, 4.0, 0.5
)

# --- Bouton ---
apply_filters = st.sidebar.button("Appliquer les filtres")

# ==============================
# MÉMORISATION DES FILTRES & PREMIER CHARGEMENT
# ==============================
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame(columns=df.columns)

if "first_run" not in st.session_state:
    st.session_state.first_run = True


def compute_filtered_df():
    filtered = df.copy()

    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]
    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]
    if selected_cuisines:
        filtered = filtered[filtered["cuisines_clean"].isin(selected_cuisines)]
    if selected_prices:
        filtered = filtered[filtered["price_level"].isin(selected_prices)]

    filtered = filtered[filtered["avg_rating"] >= min_rating]
    return filtered


# 👉 On met à jour :
# - si l'utilisateur clique sur le bouton
# - OU au tout premier chargement de la page
if apply_filters or st.session_state.first_run:
    st.session_state.filtered_df = compute_filtered_df()
    st.session_state.first_run = False

# Récupération du dernier DataFrame filtré
filtered_df = st.session_state.filtered_df


# ==============================
# 🗺️ AFFICHAGE DE LA CARTE (persistante)
# ==============================
st.markdown("### Carte interactive des restaurants filtrés")
st.markdown(f"**{len(filtered_df)} restaurants affichés** sur la carte")

# Résumé des filtres
st.markdown("#### Filtres appliqués :")
st.write(
    f"**Pays :** {', '.join(selected_countries) if selected_countries else 'Aucun'} | "
    f"**Régions :** {', '.join(selected_regions[:3]) if selected_regions else 'Aucune'} | "
    f"**Cuisines :** {', '.join(selected_cuisines[:3]) if selected_cuisines else 'Aucune'} | "
    f"**Prix :** {', '.join(selected_prices) if selected_prices else 'Aucun'} | "
    f"**Note ≥ {min_rating} ⭐**"
)

# Carte
if not filtered_df.empty:
    fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="avg_rating",
        size="total_reviews_count",
        hover_name="restaurant_name",
        hover_data={
            "city": True,
            "region": True,
            "price_level": True,
            "avg_rating": True,
            "cuisines": True,
        },
        color_continuous_scale="YlOrRd",
        zoom=4,
        height=650,
    )

    fig.update_layout(
        mapbox_style="open-street-map",   # 🔥 nouveau style
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📋 Tableau
    with st.expander("Voir les détails des restaurants filtrés"):
        st.dataframe(
            filtered_df[
                ["restaurant_name", "city", "country", "region", "price_level", "avg_rating", "cuisines"]
            ].sort_values("avg_rating", ascending=False),
            use_container_width=True,
        )
else:
    st.warning(
        "Aucun restaurant à afficher. Sélectionne des filtres puis clique sur **Appliquer les filtres**."
    )
