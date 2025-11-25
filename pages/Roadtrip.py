"""
Interface Streamlit pour planifier un road trip culinaire basé sur un itinéraire réel.
"""
from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.data_loader import load_data, resolve_tripadvisor_csv_path
from utils import geo_utils, logic

st.set_page_config(page_title="Road Trip Planner — Culinaire", layout="wide")


def load_css(path: str = "assets/style.css"):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Fichier CSS manquant : assets/style.css")


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str):
    return load_data(csv_path)


DATA_CSV = str(resolve_tripadvisor_csv_path())
RAW_DF, CITY_CENTROIDS = load_dataset(DATA_CSV)
load_css()

CITY_OPTIONS = (
    RAW_DF[["city", "country"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["country", "city"])
)
CITY_TUPLES = list(CITY_OPTIONS.itertuples(index=False, name=None))
CUISINE_SET = sorted({c for cuisines in RAW_DF["cuisine_list"].dropna() for c in cuisines})


def fmt_city(option: Tuple[str, str]) -> str:
    return f"{option[0]} — {option[1]}"


def build_city_select(label: str, options: List[Tuple[str, str]], key: str, default_index: int = 0):
    idx = min(default_index, max(0, len(options) - 1))
    return st.sidebar.selectbox(label, options=options, index=idx, format_func=fmt_city, key=key)


def sample_corridor(records: List[Dict], count: int) -> List[Dict]:
    """Sélectionne des villes réparties le long du corridor."""
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return records
    step = len(records) / count
    sampled = []
    for i in range(count):
        idx = int(round(i * step))
        idx = min(idx, len(records) - 1)
        sampled.append(records[idx])
    return sampled


st.sidebar.title("Trip Setup")
start_choice = build_city_select("Ville de départ", CITY_TUPLES, "start_city")
dest_options = [opt for opt in CITY_TUPLES if opt != start_choice]
arrival_choice = build_city_select("Ville d'arrivée", dest_options, "end_city")
remaining_for_waypoints = [opt for opt in CITY_TUPLES if opt not in {start_choice, arrival_choice}]
waypoints_choice = st.sidebar.multiselect(
    "Villes jalons (optionnel)",
    options=remaining_for_waypoints,
    format_func=fmt_city,
    key="waypoints",
)

start_date = st.sidebar.date_input("Date de départ", value=date.today())
days = st.sidebar.number_input("Durée (jours)", min_value=1, max_value=30, value=7, step=1)
budget_cap = st.sidebar.number_input("Budget max (EUR)", min_value=100, max_value=20000, value=1500, step=50)
buffer_km = st.sidebar.slider("Corridor autour de la route (km)", min_value=20, max_value=120, value=50, step=5)

st.sidebar.markdown("---")
cuisines_sel = st.sidebar.multiselect("🍽️ Cuisines préférées", CUISINE_SET, default=["French", "Italian"])
price_level = st.sidebar.selectbox("💼 Niveau de prix", ["Any", "Economy", "Moderate", "Premium"], index=2)

st.sidebar.markdown("---")
bayes_prior = st.sidebar.slider("Paramètre bayésien (m)", min_value=5, max_value=200, value=50, step=5)
rating_weight = st.sidebar.slider("Poids note vs popularité", min_value=0.3, max_value=0.9, value=0.7, step=0.05)

st.sidebar.markdown("---")
plan_btn = st.sidebar.button("Tracer l'itinéraire", use_container_width=True)


def generate_route_plan():
    scored_df = logic.compute_scores(RAW_DF, bayes_prior, rating_weight)

    def coords_for(choice: Tuple[str, str]):
        return geo_utils.get_city_coordinates(choice[0], choice[1], CITY_CENTROIDS)

    start_coords = coords_for(start_choice)
    end_coords = coords_for(arrival_choice)
    waypoint_coords = []
    for opt in waypoints_choice:
        coords = coords_for(opt)
        if coords:
            waypoint_coords.append(coords)

    if not start_coords or not end_coords:
        raise ValueError("Impossible de localiser les coordonnées des villes sélectionnées.")

    osrm_points = [start_coords] + waypoint_coords + [end_coords]
    route_info = geo_utils.fetch_osrm_route(osrm_points)

    corridor_df = geo_utils.filter_cities_along_route(CITY_CENTROIDS, route_info["coordinates"], buffer_km)
    corridor_records = geo_utils.order_cities_along_route(corridor_df)

    def base_record(city: Tuple[str, str], coords: Tuple[float, float]):
        return {
            "city": city[0],
            "country": city[1],
            "latitude": coords[0],
            "longitude": coords[1],
            "route_index": geo_utils.route_position(coords[0], coords[1], route_info["coordinates"]),
        }

    ordered = [base_record(start_choice, start_coords)]
    for opt, coords in zip(waypoints_choice, waypoint_coords):
        if coords:
            ordered.append(base_record(opt, coords))
    ordered.append(base_record(arrival_choice, end_coords))

    base_keys = {(rec["city"], rec["country"]) for rec in ordered}
    remaining = max(0, days - len(ordered))
    corridor_candidates = [rec for rec in corridor_records if (rec["city"], rec["country"]) not in base_keys]
    sampled_corridor = sample_corridor(corridor_candidates, remaining or len(corridor_candidates))
    for rec in sampled_corridor:
        key = (rec["city"], rec["country"])
        if key in base_keys:
            continue
        ordered.append(rec)
        base_keys.add(key)

    ordered = sorted(ordered, key=lambda r: r["route_index"])
    itinerary_df, restos_df, estimated_budget = logic.build_itinerary(
        scored_df, ordered, days, cuisines_sel, price_level, start_date
    )

    return {
        "route": route_info,
        "ordered_cities": ordered,
        "corridor": corridor_df,
        "itinerary": itinerary_df,
        "restaurants": restos_df,
        "budget_estimate": estimated_budget,
        "start": {"city": start_choice[0], "country": start_choice[1], "coords": start_coords},
        "end": {"city": arrival_choice[0], "country": arrival_choice[1], "coords": end_coords},
    }


if plan_btn:
    try:
        st.session_state["roadtrip_plan"] = generate_route_plan()
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state.pop("roadtrip_plan", None)
        st.error(f"Erreur lors du calcul de l'itinéraire : {exc}")

plan_data = st.session_state.get("roadtrip_plan")

st.markdown(
    """
    <div class="hero-card">
        <p class="eyebrow">Culinary Road Trip</p>
        <h1>Départ ➔ Arrivée avec vraies distances routières</h1>
        <p>Optimise un trajet gourmand grâce aux données TripAdvisor et à l'API OSRM.</p>
        <div class="chips">
            <span class="chip">Départ : {}</span>
            <span class="chip">Arrivée : {}</span>
            <span class="chip">Jours : {}</span>
            <span class="chip">Corridor : {} km</span>
        </div>
    </div>
    """.format(
        fmt_city(start_choice),
        fmt_city(arrival_choice),
        days,
        buffer_km,
    ),
    unsafe_allow_html=True,
)

if not plan_data:
    st.info("Sélectionne tes villes puis clique sur **Tracer l'itinéraire** pour récupérer un trajet OSRM.")
    st.stop()

route_info = plan_data["route"]
itinerary_df = plan_data["itinerary"]
restaurants_df = plan_data["restaurants"]

col_left, col_center, col_right = st.columns([1.4, 1.2, 1.0])
with col_left:
    st.markdown("### Statistiques de route")
    st.metric("Distance routière", f"{route_info['distance_km']:.1f} km")
    st.metric("Durée estimée", f"{route_info['duration_h']:.1f} h")
with col_center:
    st.markdown("### Corridor gastronomique")
    st.metric("Villes éligibles", f"{len(plan_data['corridor'])}")
    st.metric("Arrêts planifiés", f"{len(itinerary_df)} repas")
with col_right:
    st.markdown("### Budget")
    estimated = plan_data["budget_estimate"]
    delta = estimated - budget_cap
    st.metric("Budget estimé", f"€{estimated:,.0f}", delta=f"{'+' if delta>0 else ''}{delta:,.0f} €")
    ratio = min(1.0, estimated / budget_cap) if budget_cap else 0
    st.progress(ratio, text="Estimation vs budget max")


st.markdown("### Carte & arrêts gourmands")
route_path = [[lon, lat] for lat, lon in route_info["coordinates"]]

layers = []
layers.append(
    pdk.Layer(
        "PathLayer",
        data=[{"path": route_path}],
        get_path="path",
        get_width=5,
        get_color=[44, 130, 201],
    )
)

markers = pd.DataFrame(
    [
        {
            "name": "Départ",
            "latitude": plan_data["start"]["coords"][0],
            "longitude": plan_data["start"]["coords"][1],
            "color": [46, 204, 113],
        },
        {
            "name": "Arrivée",
            "latitude": plan_data["end"]["coords"][0],
            "longitude": plan_data["end"]["coords"][1],
            "color": [231, 76, 60],
        },
    ]
)

layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=markers,
        get_position="[longitude, latitude]",
        get_radius=7000,
        get_fill_color="color",
        pickable=True,
    )
)

if not restaurants_df.empty:
    resto_layer = pdk.Layer(
        "ScatterplotLayer",
        data=restaurants_df,
        get_position="[longitude, latitude]",
        get_radius=5000,
        get_fill_color=[255, 145, 87],
        pickable=True,
    )
    layers.append(resto_layer)

avg_lat = sum(lat for lat, _ in route_info["coordinates"]) / len(route_info["coordinates"])
avg_lon = sum(lon for _, lon in route_info["coordinates"]) / len(route_info["coordinates"])
view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=5)
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_provider="carto",
    map_style="dark",
)

with st.container():
    st.markdown("<div class='map-card'>", unsafe_allow_html=True)
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### Itinéraire jour par jour")
if itinerary_df.empty:
    st.warning("Aucun restaurant n'a pu être trouvé sur ce corridor avec les filtres actuels.")
else:
    display_df = itinerary_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv_buf = StringIO()
    itinerary_df.to_csv(csv_buf, index=False)
    ics_str = logic.make_ics(itinerary_df)
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📥 Export CSV",
            data=csv_buf.getvalue(),
            file_name="roadtrip_itinerary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "📅 Export ICS",
            data=ics_str,
            file_name="roadtrip_itinerary.ics",
            mime="text/calendar",
            use_container_width=True,
        )

st.markdown(
    "<div class='small'>Data: TripAdvisor (usage pédagogique). Routing: API OSRM publique. "
    "Scoring bayésien + diversité culinaire intégrés.</div>",
    unsafe_allow_html=True,
)
