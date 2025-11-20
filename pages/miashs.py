import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import resolve_tripadvisor_csv_path

CSV_PATH = resolve_tripadvisor_csv_path()

# ==============================
# 🔹 CONFIG
# ==============================
st.set_page_config(page_title="Carte interactive - Road Trip Culinaire", layout="wide")

# ==============================
# 🔹 CHARGEMENT OPTIMISÉ DES DONNÉES
# ==============================
@st.cache_data(ttl=3600, max_entries=1)
def load_data():
    usecols = [
        "restaurant_name", "country", "region", "city",
        "latitude", "longitude", "avg_rating", "total_reviews_count",
        "price_level", "cuisines"
    ]
    df = pd.read_csv(CSV_PATH, usecols=usecols)

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

# --- Filtre Pays ---
selected_countries = st.sidebar.multiselect(
    "Pays",
    country_list,
    default=[]  # aucun pays pré-sélectionné
)

# --- Filtre Région dépendant ---
if selected_countries:
    possible_regions = sorted(df[df["country"].isin(selected_countries)]["region"].unique().tolist())
else:
    possible_regions = sorted(df["region"].unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Région",
    options=possible_regions,
    default=[]  # aucune région par défaut
)

# --- Cuisine ---
selected_cuisines = st.sidebar.multiselect("Cuisine", cuisine_list, default=[])

# --- Prix ---
selected_prices = st.sidebar.multiselect("Prix", price_list, default=[])

# --- Note ---
min_rating = st.sidebar.slider("Note minimale", 0.0, 5.0, 4.0, 0.5)

# --- Bouton ---
apply_filters = st.sidebar.button("Appliquer les filtres")

# ==============================
# MÉMORISATION DES FILTRES (Session State)
# ==============================
if "filtered_df" not in st.session_state:
    # Par défaut, pas de restaurants affichés
    st.session_state.filtered_df = pd.DataFrame(columns=df.columns)

if apply_filters:
    filtered_df = df.copy()

    # Application conditionnelle des filtres
    if selected_countries:
        filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]
    if selected_regions:
        filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
    if selected_cuisines:
        filtered_df = filtered_df[filtered_df["cuisines_clean"].isin(selected_cuisines)]
    if selected_prices:
        filtered_df = filtered_df[filtered_df["price_level"].isin(selected_prices)]

    filtered_df = filtered_df[filtered_df["avg_rating"] >= min_rating]

    # Sauvegarde dans la session
    st.session_state.filtered_df = filtered_df

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
        mapbox_style="carto-positron",
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
    st.warning("Aucun restaurant à afficher. Sélectionne des filtres puis clique sur **Appliquer les filtres**.")
