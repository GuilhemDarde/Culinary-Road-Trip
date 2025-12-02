import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Stats & Visualisations", layout="wide")

# Charger le CSS externe (optionnel)
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ==========================
# 🔹 Navigation
# ==========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Carte 3D", "Statistiques"])


# ==========================
# 🔹 Chargement des données (cache)
# ==========================
@st.cache_data(ttl=3600)
def load_data():
    usecols = [
        "restaurant_name", "country", "city",
        "latitude", "longitude",
        "avg_rating", "total_reviews_count",
        "cuisines"
    ]

    # Téléchargement depuis Hugging Face Hub
    local_path = hf_hub_download(
        repo_id="Amoham16/dataset-resto-10k",
        repo_type="dataset",
        filename="tripadvisor_clean.csv",
    )

    # Chargement depuis Hugging Face
    df = pd.read_csv(local_path, usecols=usecols)

    # Nettoyage de base
    df = df.dropna(subset=["latitude", "longitude"])
    df["cuisines"] = df["cuisines"].fillna("Inconnue")
    df["cuisines_clean"] = df["cuisines"].apply(lambda x: x.split(",")[0].strip())
    df["city"] = df["city"].fillna("Inconnue")
    df["country"] = df["country"].fillna("Inconnu")

    return df


df = load_data()


# ===========================
# PAGE 1 : CARTE 3D
# ===========================
if page == "Carte 3D":
    st.title("Carte 3D — Visualisation des restaurants")

    # -------- Filtres ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        cuisine = st.selectbox(
            "Type de cuisine",
            ["Toutes"] + sorted(df["cuisines_clean"].unique())
        )
    
    
    with col2:
    # Option spéciale "Tous les pays"
        country_options = ["Tous les pays"] + sorted(df["country"].unique())

        selected_countries = st.multiselect(
            "Pays",
            options=country_options,
            default=["Belgium"]  # par défaut : pas de filtre sur le pays
        )
        
    with col3:
        min_rating = st.slider(
            "Note minimum", 0.0, 5.0, 4.0, step=0.1
        )

    # Appliquer filtres
    df_filtered = df.copy()

    if cuisine != "Toutes":
        df_filtered = df_filtered[df_filtered["cuisines_clean"] == cuisine]

    if selected_countries and "Tous les pays" not in selected_countries:
        df_filtered = df_filtered[df_filtered["country"].isin(selected_countries)]

    df_filtered = df_filtered[df_filtered["avg_rating"] >= min_rating]

    # -------- Mode d'affichage ----------
    st.subheader("Mode d'affichage (hauteur des colonnes)")

    mode = st.selectbox(
        "Afficher la hauteur en fonction de :",
        [
            "Popularité (note × avis)",
            "Nombre d'avis",
            "Note moyenne",
            
            "Uniforme"
        ]
    )

    df_filtered = df_filtered.copy()  # éviter les warnings

    if mode == "Nombre d'avis":
        df_filtered["height"] = df_filtered["total_reviews_count"]

    elif mode == "Note moyenne":
        df_filtered["height"] = df_filtered["avg_rating"] * 20  # scaling

    elif mode == "Popularité (note × avis)":
        df_filtered["height"] = (
            df_filtered["avg_rating"] * df_filtered["total_reviews_count"] / 5
        )

    else:  # Uniforme
        df_filtered["height"] = 50

    # Normalisation si trop grand
    if not df_filtered.empty and df_filtered["height"].max() > 5000:
        df_filtered["height"] = df_filtered["height"] / 50

    # -------- Carte 3D ----------
    st.subheader("🗺️ Carte 3D")

    if df_filtered.empty:
        st.warning("Aucun restaurant trouvé avec ces filtres.")
    else:
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
            get_elevation="height",        # 🔹 on utilise la colonne calculée
            elevation_scale=15,
            radius=12000,
            get_color=[60, 120, 255, 180],
            pickable=True,
            auto_highlight=True,
        )

        deck_3d = pdk.Deck(
            initial_view_state=view,
            layers=[layer_3d],
            tooltip={
                "text": "{restaurant_name}\n⭐ {avg_rating}\nAvis: {total_reviews_count}"
            },
        )

        st.pydeck_chart(deck_3d)

    # -------- Analyses graphiques ----------
    st.subheader("Analyses complémentaires")

    if not df_filtered.empty:
        df_country = (
            df_filtered.groupby("country")
            .agg(
                avg_rating_mean=("avg_rating", "mean"),
                total_reviews_sum=("total_reviews_count", "sum"),
                count_resto=("restaurant_name", "count"),
            )
            .sort_values(by="avg_rating_mean", ascending=False)
        )

        df_cuisine = (
            df_filtered.groupby("cuisines_clean")
            .agg(
                avg_rating_mean=("avg_rating", "mean"),
                total_reviews_sum=("total_reviews_count", "sum"),
                count_resto=("restaurant_name", "count"),
            )
            .sort_values(by="avg_rating_mean", ascending=False)
        )

        st.write("Top pays (par note moyenne) :")
        st.dataframe(df_country.head(10))

        st.write("Top cuisines (par note moyenne) :")
        st.dataframe(df_cuisine.head(10))
    else:
        st.info("Pas d'analyses possibles : aucun restaurant après filtrage.")


# ===========================
# PAGE 2 : TRENDING / STAT
# ===========================
elif page == "Statistiques":
    st.header("Statistiques")

    df_cuisine = df.copy()
    
    # ==========================
    # 2️⃣ FILTRE PAR PAYS (APPLIQUÉ APRÈS CUISINE)
    # ==========================
    st.subheader("Filtre par pays")

    country_filter = st.multiselect(
        "Filter by Cuisine",
        options=sorted(df_cuisine["cuisines_clean"].unique()),  # 👈 dépend déjà du filtre cuisine
        default=None,
    )

    # df filtré sur cuisine ⬅️ puis pays ⬅️
    filtered_df = df_cuisine.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df["cuisines_clean"].isin(country_filter)]

    # ==========================
    # METRICS (sur les deux filtres)
    # ==========================
    st.subheader("Résumé global")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Restaurants", len(filtered_df))
    with c2:
        if len(filtered_df) > 0:
            st.metric("Average Rating", f"{filtered_df['avg_rating'].mean():.2f}")
        else:
            st.metric("Average Rating", "N/A")
    with c3:
        st.metric("Countries", filtered_df["country"].nunique())

    # ==========================
    # 2e GRAPHE : Top pays (cuisine + pays)
    # ==========================
    st.subheader("Top Countries by Restaurant Count (with both filters)")
    
    if not filtered_df.empty:
        country_stats = (
            filtered_df.groupby("country")
            .agg(
                restaurant_count=("restaurant_name", "count"),
                avg_rating_mean=("avg_rating", "mean"),
            )
            .sort_values("restaurant_count", ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            country_stats,
            x="restaurant_count",
            y="country",
            orientation="h",
            labels={
                "restaurant_count": "Number of Restaurants",
                "country": "Country",
            },
            color="avg_rating_mean",             # Couleur = moyenne des notes
            color_continuous_scale="Viridis",
            hover_data={
                "avg_rating_mean": ":.2f",       # ⭐ note moyenne
                "restaurant_count": True,
                "country": False,
            },
        )

        fig.update_layout(
            height=450,
            coloraxis_colorbar_title="Avg Rating ⭐",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display for country distribution with current filters.")

    st.markdown("---")


# ==========================
    # 1️⃣ FILTRE PAR CUISINE
    # ==========================
    st.subheader("Filtre par type de cuisine")

    cuisine_filter = st.multiselect(
        "Filter by Pays",
        options=sorted(df["country"].unique()),
        default=None,
    )

    # df filtré UNIQUEMENT par pays
    
    if cuisine_filter:
        df_cuisine = df_cuisine[df_cuisine["country"].isin(cuisine_filter)]

    # ==========================
    # 1er GRAPHE : dépend SEULEMENT du filtre cuisine
    # ==========================
    st.subheader("Cuisine Type Distribution (filtered by cuisine)")

    if not df_cuisine.empty:
        cuisine_counts = df_cuisine["cuisines_clean"].value_counts()
        fig2 = px.pie(
            values=cuisine_counts.values,
            names=cuisine_counts.index,
            title="",
            hole=0.4,
        )
        fig2.update_traces(textinfo="percent",       
                           textposition="inside",  # orientation lisible
                           )
        
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data to display for cuisine distribution with current cuisine filter.")

    st.markdown("---")

