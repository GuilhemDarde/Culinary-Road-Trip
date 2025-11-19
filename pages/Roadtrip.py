# Roadtrip.py — Streamlit single-page
# Planification d'un road trip culinaire 100% offline (CSV TripAdvisor)
# Dépendances : streamlit, pandas, numpy, pydeck, scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from io import StringIO
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# =========================================================
#  CONFIG
# =========================================================
st.set_page_config(page_title="Road Trip Planner — Culinaire", layout="wide")

# >>>> Modifie ici si besoin <<<<
DATA_CSV = "/Users/aya31/Desktop/M2 MIASHS/Open data/Culinary-Road-Trip/tripadvisor_european_restaurants.csv"

# =========================================================
#  STYLE (dark UI proche du mockup fourni)
# =========================================================
DARK_CSS = """
<style>
:root {
  --bg:#0f0f0f; --panel:#181818; --muted:#222; --text:#e7e7e7; --sub:#b9b9b9; --accent:#ff6a3d; --ok:#23c55e;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
h1,h2,h3,h4 { color: var(--text); }
.sidebar .sidebar-content, [data-testid="stSidebarContent"] { background: var(--panel) !important; }
.block-container { padding-top: 1.5rem; }
hr { border-top: 1px solid var(--muted); }
.badge { display:inline-flex; gap:.4rem; align-items:center; padding:.22rem .55rem; border-radius:999px; background:#221b19; color:#ffb099; border:1px solid #3a2a25; font-size:.85rem; }
.card { background: var(--panel); border: 1px solid #2a2a2a; border-radius: 16px; padding: 1rem; }
.kpi { border-radius:14px; padding: .85rem 1rem; background:#141414; border:1px solid #232323; }
.kpi h3 { margin:.1rem 0 .3rem 0; font-size: .95rem; color:#bbb; }
.kpi p { margin:0; font-size:1.2rem; font-weight:600; }
.small { color: var(--sub); font-size:.9rem; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# =========================================================
#  DATA LOADING & CLEANING
# =========================================================
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Harmonise les colonnes principales si besoin
    colmap = {
        'name':'restaurant_name',
        'Restaurant Name':'restaurant_name',
        'lat':'latitude', 'lng':'longitude', 'lon':'longitude',
        'City':'city', 'Country':'country',
        'Rating':'rating', 'Reviews':'num_reviews',
        'Cuisine':'cuisine', 'Price':'price_level'
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    keep = [c for c in [
        'restaurant_name','address','city','country','latitude','longitude',
        'rating','num_reviews','price_level','cuisine','awards','ranking_position',
        'ranking_category','url'
    ] if c in df.columns]
    df = df[keep].copy()

    # Types numériques
    for col in ['rating','num_reviews','latitude','longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalisation prix -> numérique
    def price_to_num(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        if s and set(s) <= set("$€"):
            # "$", "$$", "$$$", "€€", etc.
            return float(sum(ch in "$€" for ch in s))
        if '-' in s:
            parts = [p.strip() for p in s.split('-')]
            nums = [sum(ch in "$€" for ch in p) for p in parts if p]
            return float(np.mean(nums)) if nums else np.nan
        lut = {'low':1,'cheap':1,'moderate':2,'mid':2,'medium':2,
               'expensive':3,'premium':3,'fine':4,'luxury':4}
        s2 = s.lower().replace('€','').replace('$','')
        return float(lut.get(s2, np.nan))
    if 'price_level' in df.columns:
        df['price_num'] = df['price_level'].apply(price_to_num)

    # Cuisines
    if 'cuisine' in df.columns:
        df['cuisine_list'] = df['cuisine'].fillna('').apply(
            lambda s: [x.strip() for x in str(s).split(',') if x.strip()])
        df['cuisine_major'] = df['cuisine_list'].apply(lambda L: L[0] if len(L)>0 else np.nan)
    else:
        df['cuisine_list'] = [[] for _ in range(len(df))]
        df['cuisine_major'] = np.nan

    # Centroïdes des villes (pour la carte & itinéraire)
    if {'country','city','latitude','longitude'}.issubset(df.columns):
        centroids = df.groupby(['country','city'])[['latitude','longitude']].mean().reset_index()
    else:
        centroids = pd.DataFrame(columns=['country','city','latitude','longitude'])

    return df, centroids

@st.cache_data
def add_scores(df: pd.DataFrame):
    """Ajoute rating_weighted, hotness_scaled et score_final."""
    df = df.copy()
    if 'rating' not in df.columns:
        df['score_final'] = 0.0
        return df

    m = 50
    C = df['rating'].dropna().mean()
    v = df['num_reviews'].fillna(0)
    R = df['rating'].fillna(C)
    df['rating_weighted'] = (v * R + m * C) / (v + m)

    mu = df.groupby('city')['rating'].transform('mean')
    sd = df.groupby('city')['rating'].transform('std').replace(0, np.nan)
    z = (df['rating'] - mu) / sd
    z = z.clip(-2, 2).fillna(0.0)

    scaler = MinMaxScaler()
    df['hotness_scaled'] = scaler.fit_transform(z.values.reshape(-1,1))

    df['score_final'] = 0.7 * df['rating_weighted'] + 0.3 * df['hotness_scaled']
    return df

# =========================================================
#  GEO & ITINERARY HELPERS
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p = np.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def nearest_neighbor_order(points):
    """points: list of (city, lat, lon, country) -> greedy route"""
    if not points: return []
    pts = points.copy()
    route = [pts.pop(0)]  # 1ère ville = départ
    while pts:
        last = route[-1]
        dists = [(haversine_km(last[1], last[2], p[1], p[2]), i) for i,p in enumerate(pts)]
        _, idx = min(dists, key=lambda x: x[0])
        route.append(pts.pop(idx))
    return route

def distance_for_route(route):
    if len(route) < 2: return 0.0
    return sum(haversine_km(route[i][1], route[i][2], route[i+1][1], route[i+1][2])
               for i in range(len(route)-1))

def pick_restaurants_for_city(df_city, cuisines_selected, price_level, n=2):
    data = df_city.copy()
    # Cuisines
    if cuisines_selected:
        data = data[data['cuisine_list'].apply(lambda L: any(c in L for c in cuisines_selected))]
    # Prix
    if price_level != "Any" and 'price_num' in data.columns:
        if price_level == "Economy":
            data = data[data['price_num'] <= 1.5]
        elif price_level == "Moderate":
            data = data[(data['price_num'] > 1.5) & (data['price_num'] <= 3.0)]
        elif price_level == "Premium":
            data = data[data['price_num'] > 3.0]

    if data.empty:
        data = df_city
    return data.sort_values('score_final', ascending=False).head(n)

def build_itinerary(cities_ordered, df, days, cuisines, price_level):
    """
    Returns (itinerary_df, restaurants_df).
    itinerary_df: day, city, country, lunch_name, dinner_name
    restaurants_df: points pour la carte
    """
    rows, restos = [], []
    if not cities_ordered:
        return pd.DataFrame(columns=['day','city','country','lunch_name','dinner_name']), pd.DataFrame()

    n_cities = len(cities_ordered)
    base, rem = days // n_cities, days % n_cities
    days_per_city = [base + (1 if i < rem else 0) for i in range(n_cities)]

    current_day = 1
    for i, (city, lat, lon, country) in enumerate(cities_ordered):
        sub = df[(df['city']==city) & (df['country']==country)]
        stay = max(1, days_per_city[i])
        for _ in range(stay):
            picks = pick_restaurants_for_city(sub, cuisines, price_level, n=4)
            lunch  = picks.iloc[0] if len(picks)>=1 else None
            dinner = picks.iloc[1] if len(picks)>=2 else None

            rows.append({
                'day': current_day,
                'city': city,
                'country': country,
                'lunch_name': lunch['restaurant_name'] if lunch is not None else '',
                'dinner_name': dinner['restaurant_name'] if dinner is not None else '',
            })

            for tag, row in [('Restaurants', lunch), ('Restaurants', dinner)]:
                if row is not None:
                    restos.append({
                        'type': tag,
                        'restaurant_name': row['restaurant_name'],
                        'city': city, 'country': country,
                        'latitude': row['latitude'], 'longitude': row['longitude'],
                        'rating': row.get('rating', np.nan),
                        'price_level': row.get('price_level', ''),
                        'cuisine': row.get('cuisine', '')
                    })

            current_day += 1
            if current_day > days: break
        if current_day > days: break

    return pd.DataFrame(rows), pd.DataFrame(restos)

def make_ics(itinerary_df, start_date=None):
    if start_date is None:
        start_date = datetime.today().date()
    ics = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//RoadTripCulinaire//EN"]
    for _, row in itinerary_df.iterrows():
        day = int(row['day'])
        d0 = datetime.combine(start_date + timedelta(days=day-1), datetime.min.time())
        for when, title in [('12:30','Lunch'), ('19:30','Dinner')]:
            h, m = map(int, when.split(':'))
            dtstart = d0 + timedelta(hours=h, minutes=m)
            dtend = dtstart + timedelta(hours=1, minutes=30)
            summary = f"{title} — {row['city']} — {row[title.lower()+'_name']}"
            ics += [
                "BEGIN:VEVENT",
                f"DTSTART:{dtstart.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{dtend.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:{summary}",
                "END:VEVENT"
            ]
    ics.append("END:VCALENDAR")
    return "\n".join(ics)

# =========================================================
#  LOAD DATA
# =========================================================
DF, CITY_CENTROIDS = load_data(DATA_CSV)
DF = add_scores(DF)

# =========================================================
#  SIDEBAR — Trip Setup
# =========================================================
st.sidebar.title("Trip Setup")
st.sidebar.markdown("_Plan your journey_")

# Cuisines
all_cuisines = sorted(set(sum(DF['cuisine_list'].dropna().tolist(), [])))
default_cuis = [c for c in ["French","Italian"] if c in all_cuisines]
cuisines_sel = st.sidebar.multiselect("🍽️ Food Preferences", all_cuisines[:150], default=default_cuis)

# Pays & Villes
countries = sorted(DF['country'].dropna().unique().tolist())
country_sel = st.sidebar.selectbox("🌍 Country", countries) if countries else None

cities = sorted(DF[DF['country']==country_sel]['city'].dropna().unique().tolist()) if country_sel else []
cities_sel = st.sidebar.multiselect("🏙️ Cities (add 2+)", cities, default=[])

# Durée & Budget
days = st.sidebar.number_input("📅 Days", min_value=1, max_value=60, value=7, step=1)
budget_day = st.sidebar.number_input("💶 Daily budget (EUR)", min_value=10, max_value=2000, value=150, step=10)
price_bucket = st.sidebar.selectbox("💼 Price level", ["Any","Economy","Moderate","Premium"], index=2)

st.sidebar.markdown("---")
optimize_btn = st.sidebar.button("🧭 Optimize Route", use_container_width=True)

# =========================================================
#  HEADER / KPIs
# =========================================================
colL, colC, colR = st.columns([1.2, 2.2, 1.2])
with colL:
    st.markdown("## 🧭 Road Trip Planner")
    st.caption("Discover Europe’s finest culinary destinations")
    if not cities_sel:
        st.info("👉 Ajoute au moins **2 villes** pour planifier l’itinéraire.")
    elif len(cities_sel) == 1:
        st.info("👉 Ajoute encore **1 ville** pour faire un parcours.")
    else:
        st.success(f"{len(cities_sel)} ville(s) sélectionnée(s) — {country_sel}")

with colR:
    st.markdown("## Summary")
    k1, k2 = st.columns(2); k3, k4 = st.columns(2)
    with k1: st.markdown('<div class="kpi"><h3>Distance</h3><p id="kpi-dist">—</p></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi"><h3>Cost</h3><p>€{budget_day*days:,}</p></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi"><h3>Duration</h3><p>{days} days</p></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi"><h3>Avg/day</h3><p>€{budget_day}</p></div>', unsafe_allow_html=True)

with colC:
    st.markdown("## Map & Route")

# =========================================================
#  ROUTE BUILD
# =========================================================
route_points = []
if cities_sel:
    # Centroïdes des villes choisies
    candidates = CITY_CENTROIDS[(CITY_CENTROIDS['country']==country_sel) &
                                (CITY_CENTROIDS['city'].isin(cities_sel))]
    if candidates.empty:
        tmp = DF[(DF['country']==country_sel) & (DF['city'].isin(cities_sel))] \
            .groupby(['country','city'])[['latitude','longitude']].mean().reset_index()
        candidates = tmp

    points = [(row['city'], row['latitude'], row['longitude'], country_sel)
              for _, row in candidates.sort_values('city').iterrows()]

    if len(points) >= 2:
        if optimize_btn:
            ordered = nearest_neighbor_order(points)
            st.session_state['ordered_cities'] = ordered
        else:
            ordered = st.session_state.get('ordered_cities', points)
        route_points = ordered

# =========================================================
#  ITINERARY + MAP
# =========================================================
itinerary_df = pd.DataFrame()
selected_restos = pd.DataFrame()
total_dist = 0.0

if route_points:
    total_dist = distance_for_route(route_points)
    itinerary_df, selected_restos = build_itinerary(route_points, DF, days, cuisines_sel, price_bucket)

    # Map layers
    path_coords = [[p[2], p[1]] for p in route_points]  # [lon, lat]
    cities_df_map = pd.DataFrame({
        'name':[p[0] for p in route_points],
        'country':[p[3] for p in route_points],
        'latitude':[p[1] for p in route_points],
        'longitude':[p[2] for p in route_points],
        'type':['Start'] + (['Stops']*(len(route_points)-2) if len(route_points)>2 else []) + (['End'] if len(route_points)>1 else [])
    })

    layers = []
    if len(path_coords) >= 2:
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": path_coords, "name": "route"}],
            get_path="path",
            width_scale=2,
            width_min_pixels=4,
            get_color=[255, 106, 61, 130],
            get_width=4,
        ))

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=cities_df_map,
        get_position='[longitude, latitude]',
        get_radius=6000,
        pickable=True,
        get_fill_color=[255, 106, 61, 220],
    ))

    if not selected_restos.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=selected_restos,
            get_position='[longitude, latitude]',
            get_radius=3000,
            pickable=True,
            get_fill_color=[180, 180, 255, 180],
        ))

    initial_view = pdk.ViewState(
        latitude=np.mean([p[1] for p in route_points]),
        longitude=np.mean([p[2] for p in route_points]),
        zoom=5
    )

    r = pdk.Deck(layers=layers, initial_view_state=initial_view)
    st.pydeck_chart(r, use_container_width=True)

    with colR:
        st.markdown(f'<div class="kpi"><h3>Distance</h3><p>{total_dist:.0f} km</p></div>', unsafe_allow_html=True)

# =========================================================
#  DETAILS + EXPORTS
# =========================================================
st.markdown("### Itinerary (day by day)")
if itinerary_df.empty:
    st.caption("— en attente d’une sélection de villes et de l’optimisation de l’itinéraire —")
else:
    st.dataframe(itinerary_df, use_container_width=True, height=360)

if not itinerary_df.empty:
    # CSV
    csv_buf = StringIO()
    itinerary_df.to_csv(csv_buf, index=False)
    st.download_button("📥 Export CSV", csv_buf.getvalue(), "itinerary.csv",
                       mime="text/csv", use_container_width=True)

    # ICS
    ics_str = make_ics(itinerary_df)
    st.download_button("📅 Export ICS (calendar)", ics_str, "itinerary.ics",
                       mime="text/calendar", use_container_width=True)

# =========================================================
#  FOOTER
# =========================================================
st.markdown("<div class='small'>Data: CSV TripAdvisor (usage pédagogique). Carto: pydeck/Deck.gl. Aucune API externe requise.</div>", unsafe_allow_html=True)