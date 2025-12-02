import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from huggingface_hub import hf_hub_download

# ===========================
# 🔹 Chargement & préparation des données
# ===========================
@st.cache_data(ttl=3600)
def load_data():
    """Load and clean the European restaurants dataset"""
    usecols = [
        "restaurant_name", "country", "region", "province", "city",
        "address", "latitude", "longitude",
        "price_level", "price_range",
        "cuisines",
        "avg_rating", "total_reviews_count"
    ]
    # Téléchargement depuis Hugging Face Hub
    local_path = hf_hub_download(
        repo_id="Amoham16/dataset-resto-10k",
        repo_type="dataset",
        filename="tripadvisor_clean.csv",
    )

    # Chargement depuis Hugging Face
    df = pd.read_csv(local_path, usecols=usecols)


    # Nettoyage des colonnes texte
    text_cols = ["country", "region", "province", "city", "address", "price_level", "price_range", "cuisines"]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("Inconnu")
            .str.strip()
            .replace("", "Inconnu")
        )

    # Colonnes numériques
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["total_reviews_count"] = pd.to_numeric(df["total_reviews_count"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # On enlève les lignes sans coordonnées ou sans note
    df = df.dropna(subset=["latitude", "longitude", "avg_rating"])

    # Colonne "cuisine principale" (première cuisine de la liste)
    def first_cuisine(x):
        if x is None or pd.isna(x):
            return "Inconnue"
        parts = str(x).split(",")
        return parts[0].strip() if parts else "Inconnue"

    df["cuisine_main"] = df["cuisines"].apply(first_cuisine)

    return df


df = load_data()

# ===========================
# 🔹 Titre principal
# ===========================
st.title("Open Data Culinary Road Trip")
st.markdown("*Discover trending restaurants, find the best spots nearby, and plan a foodie road trip across Europe!*")
st.write("Bienvenue dans votre exploration culinaire en Europe à partir de données **Open Data** ")

# ===========================
# PAGE : ROAD TRIP PLANNER
# ===========================
st.header("Multi-Day Foodie Road Trip Planner")
st.markdown("Plan your perfect culinary journey across Europe!")

# --- stocker le résultat dans le state (simple) ---
if "roadtrip_results" not in st.session_state:
    st.session_state.roadtrip_results = None

# =========
# CRITÈRES
# =========
c1, c2 = st.columns(2)

with c1:
    st.subheader("Trip Settings")
    min_rating_trip = st.slider(
        "Minimum Restaurant Rating",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
    )

    preferred_cuisines = st.multiselect(
        "Preferred Cuisines (optional)",
        options=sorted(df["cuisine_main"].unique()),
        default=None,
    )

with c2:
    st.subheader("Location Preferences")

    all_countries = sorted(df["country"].unique())

    # 👉 Par défaut : France + Italy (si présents dans le dataset)
    default_countries = [c for c in ["France", "Italy"] if c in all_countries]

    preferred_countries = st.multiselect(
        "Countries to Visit",
        options=all_countries,
        default=default_countries,
    )

    # 🔍 Villes selon les pays sélectionnés
    if preferred_countries:
        base_df = df[df["country"].isin(preferred_countries)].copy()
        possible_cities = sorted(base_df["city"].unique())
    else:
        base_df = df.copy()
        possible_cities = []

    # 👉 Ne pré-sélectionner des villes QUE si des pays sont choisis
    if preferred_countries:
        default_cities = possible_cities[:3] if len(possible_cities) >= 3 else possible_cities
    else:
        default_cities = []

    selected_cities = st.multiselect(
        "Cities to include in the trip",
        options=possible_cities,
        default=default_cities,
    )

    # Affichage explicatif
    if preferred_countries:
        st.caption(f"{len(possible_cities)} ville(s) trouvée(s) pour les pays sélectionnés.")
    else:
        st.caption("Sélectionne d'abord au moins un pays pour voir les villes disponibles.")

# ==========================
# 🔹 Nombre de jours par ville
# ==========================
st.markdown("### Number of days per city")

days_per_city = {}
total_days = 0

for city in selected_cities:
    days = st.number_input(
        f"Days in {city}",
        min_value=1,
        max_value=30,
        value=2,
        step=1,
        key=f"days_{city}",
    )
    days_per_city[city] = days
    total_days += days

# ======================
#   GÉNÉRATION DU TRIP
# ======================
if st.button("🎯 Generate Road Trip", type="primary"):
    if not selected_cities:
        st.warning("Please select at least one city.")
        st.session_state.roadtrip_results = None
    else:
        trip_df = df.copy()

        # Appliquer les mêmes filtres que ceux utilisés pour la sélection
        trip_df = trip_df[trip_df["avg_rating"] >= min_rating_trip]

        if preferred_cuisines:
            trip_df = trip_df[trip_df["cuisine_main"].isin(preferred_cuisines)]

        if preferred_countries:
            trip_df = trip_df[trip_df["country"].isin(preferred_countries)]

        if selected_cities:
            trip_df = trip_df[trip_df["city"].isin(selected_cities)]

        # Coût estimé (optionnel) à partir de price_level / price_range
        price_to_cost = {
            "€": 20,
            "€€": 40,
            "€€€": 80,
            "€€€€": 150,
        }
        trip_df["estimated_cost"] = trip_df["price_level"].map(price_to_cost)

        selected_restaurants = []

        for city, n_days in days_per_city.items():
            city_df = trip_df[trip_df["city"] == city].sort_values(
                ["avg_rating", "total_reviews_count"],
                ascending=[False, False],
            )

            if len(city_df) == 0:
                st.warning(f"No restaurants found for {city} with these filters.")
                continue

            if len(city_df) < n_days:
                st.warning(
                    f"Only {len(city_df)} restaurants found for {city} (need {n_days})."
                )
                n_days = len(city_df)

            # 1 resto par jour : top n_days
            selected_restaurants.extend(
                city_df.head(n_days).to_dict(orient="records")
            )

        if len(selected_restaurants) == 0:
            st.warning(
                "No restaurants found. Try relaxing your filters (rating, cuisine, etc.)."
            )
            st.session_state.roadtrip_results = None
        else:
            total_cost = float(
                sum(
                    r.get("estimated_cost", 0)
                    for r in selected_restaurants
                    if not pd.isna(r.get("estimated_cost", np.nan))
                )
            )
            countries_visited = len(set(r["country"] for r in selected_restaurants))
            avg_rating_trip = float(np.mean([r["avg_rating"] for r in selected_restaurants]))

            st.session_state.roadtrip_results = {
                "selected_restaurants": selected_restaurants,
                "days_per_city": days_per_city,
                "total_days": total_days,
                "countries_visited": countries_visited,
                "avg_rating": avg_rating_trip,
                "total_cost": total_cost,
            }

            st.success("✅ Road trip generated and saved! Scroll to see it below 👇")

# ======================
#   AFFICHAGE DU TRIP
# ======================
results = st.session_state.roadtrip_results

if results is None:
    st.info(
        "Choose countries, cities and days per city, then click **Generate Road Trip**."
    )
else:
    selected_restaurants = results["selected_restaurants"]
    days_per_city = results["days_per_city"]
    total_days = results["total_days"]

    st.subheader("📊 Trip Summary")
    cc1, cc2, cc3, cc4 = st.columns(4)

    with cc1:
        st.metric("Total Days", total_days)
    with cc2:
        st.metric("Restaurant Stops", len(selected_restaurants))
    with cc3:
        st.metric("Countries", results["countries_visited"])
    with cc4:
        st.metric("Avg Rating", f"⭐ {results['avg_rating']:.2f}")

    st.subheader("Your Itinerary")

    day_idx = 0
    for city, n_days in days_per_city.items():
        if n_days == 0:
            continue

        st.markdown(f"## 📍 {city} — {n_days} day(s)")

        for _ in range(n_days):
            if day_idx >= len(selected_restaurants):
                break

            r = selected_restaurants[day_idx]

            st.markdown(f"### Day {day_idx + 1}")
            with st.container():
                k1, k2, k3 = st.columns([3, 1, 1])

                with k1:
                    st.markdown(f"**Restaurant: {r['restaurant_name']}**")
                    st.markdown(f"📍 {r['city']}, {r['country']} | 🍴 {r['cuisines']}")
                    st.markdown(
                        f"📧 {r.get('address', 'Address not available')}"
                    )

                with k2:
                    st.markdown(f"⭐ **{r['avg_rating']}**")
                    if not pd.isna(r.get("estimated_cost", np.nan)):
                        st.markdown(
                            f"💰 {r['price_level']} (~€{int(r['estimated_cost'])})"
                        )
                    else:
                        st.markdown(f"💰 {r['price_level']}")

                    st.markdown(f"🧾 {int(r['total_reviews_count'])} reviews")

                with k3:
                    st.markdown(f"📞 {r.get('phone', 'N/A')}")  # dataset n'a pas phone, donc N/A

            day_idx += 1

        st.divider()

    # ======================
    #        MAP
    # ======================
    st.subheader("🗺️ Trip Map")

    trip_restaurants_df = pd.DataFrame(selected_restaurants)
    center_lat = trip_restaurants_df["latitude"].mean()
    center_lon = trip_restaurants_df["longitude"].mean()

    trip_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    for idx, r in enumerate(selected_restaurants):
        popup_html = f"""
        <b>Stop {idx+1}: {r['restaurant_name']}</b><br>
        {r['city']}, {r['country']}<br>
        ⭐ {r['avg_rating']} ({int(r['total_reviews_count'])} reviews)<br>
        🍴 {r['cuisines']}<br>
        💰 {r['price_level']}
        """
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=popup_html,
            tooltip=f"Stop {idx+1}: {r['restaurant_name']}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background-color: #FF4B4B;
                    color: white;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 14px;
                    border: 2px solid white;">
                    {idx+1}
                </div>
                """
            ),
        ).add_to(trip_map)

    coordinates = [
        [r["latitude"], r["longitude"]] for r in selected_restaurants
    ]

    if len(coordinates) >= 2:
        folium.PolyLine(
            coordinates,
            color="#FF4B4B",
            weight=3,
            opacity=0.7,
            popup="Trip Route",
        ).add_to(trip_map)

    st_folium(trip_map, width=1400, height=500)
