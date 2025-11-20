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

from utils.data_loader import resolve_tripadvisor_csv_path

# =========================================================
#  CONFIG
# =========================================================
st.set_page_config(page_title="Road Trip Planner — Culinaire", layout="wide")

# >>>> Modifie ici si besoin <<<<
DATA_CSV = resolve_tripadvisor_csv_path()

# =========================================================
#  STYLE (dark UI proche du mockup fourni)
# =========================================================
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
:root {
  --bg:#040810;
  --bg-soft:#0a1326;
  --panel:rgba(13,18,33,0.92);
  --panel-border:rgba(255,255,255,0.08);
  --text:#f4f6fb;
  --sub:#96a1b5;
  --accent:#ff9457;
  --accent-strong:#ff713d;
  --success:#25d0a2;
  --warning:#ffce54;
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 10% 20%, #152042 0%, #050914 50%, #03060d 100%);
  color: var(--text);
  font-family: "Space Grotesk", "Inter", sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.sidebar .sidebar-content, [data-testid="stSidebarContent"] {
  background: var(--bg-soft) !important;
  border-right: 1px solid rgba(255,255,255,0.05);
}
.sidebar .stSelectbox label,
.sidebar .stMultiselect label,
.sidebar .stNumberInput label {
  color: var(--text);
  font-weight: 500;
}
.sidebar input, .sidebar textarea {
  background: rgba(255,255,255,0.05);
  color: var(--text);
}
.sidebar .stButton button {
  background: var(--accent);
  color: #0b0b12;
  font-weight: 600;
  border-radius: 14px;
  height: 3rem;
  border: none;
  box-shadow: 0 8px 24px rgba(255,148,87,0.35);
}
.block-container { padding-top: 1.2rem; }
.hero-card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 26px;
  padding: 1.7rem;
  box-shadow: 0 25px 70px rgba(5,9,19,0.65);
}
.hero-card h1 { margin: 0 0 .7rem 0; }
.hero-card p { margin: 0 0 .9rem 0; color: var(--sub); }
.eyebrow { letter-spacing: .35em; text-transform: uppercase; font-size: .75rem; color: var(--sub); margin-bottom: .45rem; }
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 1rem; }
.chip {
  border-radius: 999px;
  border: 1px solid var(--panel-border);
  padding: .35rem .85rem;
  font-size: .85rem;
  color: var(--text);
  background: rgba(255,255,255,0.04);
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  border-radius: 14px;
  padding: .6rem 1rem;
  font-weight: 600;
  font-size: .9rem;
}
.status-pill.ok {
  background: rgba(37,208,162,0.15);
  color: #4df4c8;
  border: 1px solid rgba(37,208,162,0.4);
}
.status-pill.warn {
  background: rgba(255,206,84,0.18);
  color: #ffde8a;
  border: 1px solid rgba(255,206,84,0.4);
}
.section-card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 24px;
  padding: 1.2rem 1.5rem;
  margin-top: 1.2rem;
  box-shadow: 0 18px 45px rgba(4,8,16,0.6);
}
.section-card h3 { margin-top: 0; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .9rem;
  margin-top: .8rem;
}
.summary-grid .kpi {
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 1rem;
  border: 1px solid var(--panel-border);
}
.summary-grid .kpi h3 { font-size: .85rem; color: var(--sub); margin-bottom: .3rem; }
.summary-grid .kpi p { font-size: 1.6rem; margin: 0; font-weight: 600; }
.map-card {
  padding: 1rem;
  border-radius: 24px;
  background: var(--panel);
  border: 1px solid var(--panel-border);
  box-shadow: 0 24px 45px rgba(3,6,13,0.7);
  margin-top: 1rem;
}
div[data-testid="stDeckGlJsonChart"] {
  background: transparent;
}
div[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  padding: .5rem;
}
.stDownloadButton button {
  border-radius: 14px;
  border: none;
  background: var(--accent-strong);
  color: #0b0b12;
  font-weight: 600;
  box-shadow: 0 10px 30px rgba(255,113,61,0.35);
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# =========================================================
#  DATA LOADING & CLEANING
# =========================================================
@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)

    # Harmonise les colonnes principales si besoin
    colmap = {
        'name':'restaurant_name',
        'Restaurant Name':'restaurant_name',
        'lat':'latitude', 'lng':'longitude', 'lon':'longitude',
        'City':'city', 'Country':'country',
        'Rating':'rating', 'Reviews':'num_reviews',
        'Cuisine':'cuisine', 'Cuisines':'cuisine', 'cuisines':'cuisine',
        'Price':'price_level'
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
    def parse_cuisine(value):
        if pd.isna(value):
            return tuple()
        parts = [x.strip() for x in str(value).split(',') if x.strip()]
        return tuple(parts)

    if 'cuisine' in df.columns:
        df['cuisine_list'] = df['cuisine'].apply(parse_cuisine)
        df['cuisine_major'] = df['cuisine_list'].apply(lambda L: L[0] if len(L)>0 else np.nan)
    else:
        df['cuisine_list'] = [tuple() for _ in range(len(df))]
        df['cuisine_major'] = np.nan

    # Centroïdes des villes (pour la carte & itinéraire)
    if {'country','city','latitude','longitude'}.issubset(df.columns):
        centroids = df.groupby(['country','city'])[['latitude','longitude']].mean().reset_index()
    else:
        centroids = pd.DataFrame(columns=['country','city','latitude','longitude'])

    return df, centroids

@st.cache_data
def add_scores(df: pd.DataFrame, bayes_weight: int, rating_importance: float):
    """Ajoute rating_weighted, hotness_scaled et score_final selon les hyperparamètres."""
    df = df.copy()
    if 'rating' not in df.columns:
        df['score_final'] = 0.0
        return df

    m = max(1, bayes_weight)
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

    rating_weight = min(max(rating_importance, 0.0), 1.0)
    popularity_weight = 1.0 - rating_weight
    df['score_final'] = rating_weight * df['rating_weighted'] + popularity_weight * df['hotness_scaled']
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

def two_opt_route(route):
    """Affinage 2-opt pour réduire les croisements et la distance totale."""
    if len(route) < 4:
        return route

    best = route[:]
    improved = True

    def swap_2opt(path, i, k):
        return path[:i] + path[i:k][::-1] + path[k:]

    best_distance = distance_for_route(best)
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best)):
                if k - i == 1:
                    continue
                new_route = swap_2opt(best, i, k)
                new_distance = distance_for_route(new_route)
                if new_distance + 1e-6 < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return best

def distance_for_route(route):
    if len(route) < 2: return 0.0
    return sum(haversine_km(route[i][1], route[i][2], route[i+1][1], route[i+1][2])
               for i in range(len(route)-1))

def pick_restaurants_for_city(df_city, cuisines_selected, price_level, used_restaurants=None, n=2):
    data = df_city.copy()
    filtered = data
    # Cuisines
    if cuisines_selected:
        filtered = filtered[filtered['cuisine_list'].apply(lambda L: any(c in L for c in cuisines_selected))]
    # Prix
    if price_level != "Any" and 'price_num' in filtered.columns:
        if price_level == "Economy":
            filtered = filtered[filtered['price_num'] <= 1.5]
        elif price_level == "Moderate":
            filtered = filtered[(filtered['price_num'] > 1.5) & (filtered['price_num'] <= 3.0)]
        elif price_level == "Premium":
            filtered = filtered[filtered['price_num'] > 3.0]

    if filtered.empty:
        filtered = data

    ordered = filtered.sort_values('score_final', ascending=False)

    if used_restaurants:
        ordered = ordered[~ordered['restaurant_name'].isin(used_restaurants)]

    if ordered.empty:
        ordered = filtered.sort_values('score_final', ascending=False)

    return ordered.head(n)

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
    used_restaurants = set()
    for i, (city, lat, lon, country) in enumerate(cities_ordered):
        sub = df[(df['city']==city) & (df['country']==country)]
        stay = max(1, days_per_city[i])
        for _ in range(stay):
            picks = pick_restaurants_for_city(sub, cuisines, price_level, used_restaurants=used_restaurants, n=4)
            fallback_text = "Aucun restaurant correspondant — élargis les filtres." if picks.empty else ""
            lunch  = picks.iloc[0] if len(picks)>=1 else None
            dinner = picks.iloc[1] if len(picks)>=2 else None

            rows.append({
                'day': current_day,
                'city': city,
                'country': country,
                'lunch_name': lunch['restaurant_name'] if lunch is not None else fallback_text,
                'dinner_name': dinner['restaurant_name'] if dinner is not None else fallback_text,
            })

            for tag, row in [('Restaurants', lunch), ('Restaurants', dinner)]:
                if row is not None:
                    used_restaurants.add(row['restaurant_name'])
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

def preview_list(values, empty_label="—", limit=3):
    """Retourne une version courte d'une liste pour les puces UI."""
    if not values:
        return empty_label
    snippet = ", ".join(values[:limit])
    if len(values) > limit:
        snippet += "..."
    return snippet

def format_currency(value: float) -> str:
    """Formate un montant en euros avec séparateur fin."""
    return f"€{int(round(value)):,}".replace(",", " ")

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
#  LOAD DATA & SCORING CONTROLS
# =========================================================
BASE_DF, CITY_CENTROIDS = load_data(DATA_CSV)

st.sidebar.subheader("Scoring Preferences")
bayes_prior = st.sidebar.slider("Reliability threshold (m)", min_value=5, max_value=200, value=50, step=5,
                                help="A higher value donne plus d'importance au score moyen global lorsque peu d'avis sont disponibles.")
rating_emphasis = st.sidebar.slider("Weight: Rating vs Popularity", min_value=0.3, max_value=0.9, value=0.7, step=0.05,
                                    help="Ajuste la pondération entre la note moyenne et la popularité (Z-score des villes).")

DF = add_scores(BASE_DF, bayes_prior, rating_emphasis)
CITY_COUNT_LOOKUP = {}
for _, row in DF.groupby(['country','city'])['restaurant_name'].count().reset_index(name='count').iterrows():
    CITY_COUNT_LOOKUP.setdefault(row['country'], {})[row['city']] = int(row['count'])

# =========================================================
#  SIDEBAR — Trip Setup
# =========================================================
st.sidebar.title("Trip Setup")
st.sidebar.markdown("_Plan your journey_")

# Cuisines
all_cuisines = sorted({c for cuisines in DF['cuisine_list'].dropna() for c in cuisines if c})
default_cuis = [c for c in ["French","Italian"] if c in all_cuisines]
cuisines_sel = st.sidebar.multiselect("🍽️ Food Preferences", all_cuisines[:150], default=default_cuis)

# Pays & Villes
countries = sorted(DF['country'].dropna().unique().tolist())
country_sel = st.sidebar.selectbox("🌍 Country", countries) if countries else None

city_counts = CITY_COUNT_LOOKUP.get(country_sel, {}) if country_sel else {}
if country_sel:
    st.sidebar.markdown("##### City Finder")
    city_query = st.sidebar.text_input("🔎 Search", placeholder="Vienna, Salzburg...", key="city_search")
    prioritize_popular = st.sidebar.checkbox("Highlight foodie hotspots", value=True,
                                             help="Trie les villes par nombre de restaurants avant l'ordre alphabétique.")
    available_cities = list(city_counts.keys())
    if prioritize_popular:
        available_cities.sort(key=lambda c: city_counts.get(c, 0), reverse=True)
    else:
        available_cities.sort()
    if city_query:
        q = city_query.strip().lower()
        available_cities = [c for c in available_cities if q in c.lower()]
    if not available_cities:
        st.sidebar.caption("Aucune ville ne correspond à la recherche actuelle.")
else:
    available_cities = []
    prioritize_popular = False
    city_query = ""

def city_format(city_name: str) -> str:
    return f"{city_name} · {city_counts.get(city_name, 0)} spots"

cities_sel = st.sidebar.multiselect(
    "🏙️ Cities (add 2+)",
    available_cities,
    default=[],
    format_func=city_format if city_counts else None,
)

# Durée & Budget
days = st.sidebar.number_input("📅 Days", min_value=1, max_value=60, value=7, step=1)
budget_day = st.sidebar.number_input("💶 Daily budget (EUR)", min_value=10, max_value=2000, value=150, step=10)
price_bucket = st.sidebar.selectbox("💼 Price level", ["Any","Economy","Moderate","Premium"], index=2)

st.sidebar.markdown("---")
optimize_btn = st.sidebar.button("🧭 Optimize Route", use_container_width=True)

# =========================================================
#  ROUTE BUILD & ITINERARY PREP
# =========================================================
route_points = []
itinerary_df = pd.DataFrame()
selected_restos = pd.DataFrame()
total_dist = 0.0

if cities_sel:
    candidates = CITY_CENTROIDS[(CITY_CENTROIDS['country']==country_sel) &
                                (CITY_CENTROIDS['city'].isin(cities_sel))]
    if candidates.empty:
        tmp = DF[(DF['country']==country_sel) & (DF['city'].isin(cities_sel))] \
            .groupby(['country','city'])[['latitude','longitude']].mean().reset_index()
        candidates = tmp

    points = [(row['city'], row['latitude'], row['longitude'], country_sel)
              for _, row in candidates.sort_values('city').iterrows()]

    if len(points) >= 2:
        selection_signature = (country_sel, tuple(sorted(cities_sel)))
        needs_recompute = optimize_btn or st.session_state.get('route_signature') != selection_signature
        if needs_recompute:
            ordered = nearest_neighbor_order(points)
            optimized_route = two_opt_route(ordered)
            st.session_state['ordered_cities'] = optimized_route
            st.session_state['route_signature'] = selection_signature
        route_points = st.session_state.get('ordered_cities', two_opt_route(points))

if route_points:
    total_dist = distance_for_route(route_points)
    itinerary_df, selected_restos = build_itinerary(route_points, DF, days, cuisines_sel, price_bucket)

# =========================================================
#  HERO / SUMMARY
# =========================================================
status_class = "warn"
if not cities_sel:
    status_msg = "Ajoute au moins deux villes pour planifier ton road trip."
elif len(cities_sel) == 1:
    status_msg = "Ajoute encore une ville pour calculer l’itinéraire."
elif not route_points:
    status_msg = "Coordonnées manquantes pour ces villes. Essaie une autre sélection."
else:
    status_class = "ok"
    status_msg = f"Itinéraire optimisé sur {len(route_points)} villes (2-opt)."

country_chip = country_sel if country_sel else "Choisis un pays"
city_chip = preview_list(cities_sel, empty_label="Ajoute des villes")
cuisine_chip = preview_list(cuisines_sel, empty_label="Toutes cuisines")
score_chip = f"Scoring: {int(rating_emphasis*100)}% rating / {int((1-rating_emphasis)*100)}% pop."
distance_display = f"{total_dist:.0f} km" if total_dist else "—"
cost_display = format_currency(budget_day * days)
duration_display = f"{days} days"
avg_display = format_currency(budget_day)

hero_left, hero_right = st.columns([2.5, 1.4])
with hero_left:
    st.markdown(
        f"""
        <div class="hero-card">
            <p class="eyebrow">Road Trip Planner</p>
            <h1>Map & Route</h1>
            <p>Discover Europe’s finest culinary destinations with a curated itinerary.</p>
            <div class="chips">
                <span class="chip">Country: {country_chip}</span>
                <span class="chip">Cities: {city_chip}</span>
                <span class="chip">Cuisines: {cuisine_chip}</span>
                <span class="chip">Plan: {days} days · €{budget_day}/day</span>
                <span class="chip">{score_chip}</span>
            </div>
            <div class="status-pill {status_class}">{status_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>Summary</h3>
            <div class="summary-grid">
                <div class="kpi"><h3>Distance</h3><p>{distance_display}</p></div>
                <div class="kpi"><h3>Cost</h3><p>{cost_display}</p></div>
                <div class="kpi"><h3>Duration</h3><p>{duration_display}</p></div>
                <div class="kpi"><h3>Avg/day</h3><p>{avg_display}</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
#  MAP & ROUTE
# =========================================================
st.markdown("### Map & Route")
st.caption("Itinéraire optimisé via heuristique nearest-neighbor + 2-opt pour limiter les détours.")
if route_points:
    path_coords = [[p[2], p[1]] for p in route_points]
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
    with st.container():
        st.markdown("<div class='map-card'>", unsafe_allow_html=True)
        st.pydeck_chart(r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class='section-card'>
            <p>Sélectionne un pays puis au moins deux villes pour afficher l’itinéraire et la carte interactive.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
#  DETAILS + EXPORTS
# =========================================================
st.markdown("### Itinerary (day by day)")
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    if itinerary_df.empty:
        st.caption("— en attente d’une sélection de villes et de l’optimisation de l’itinéraire —")
    else:
        st.dataframe(itinerary_df, use_container_width=True, height=360)

        csv_buf = StringIO()
        itinerary_df.to_csv(csv_buf, index=False)

        ics_str = make_ics(itinerary_df)

        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button("📥 Export CSV", csv_buf.getvalue(), "itinerary.csv",
                               mime="text/csv", use_container_width=True)
        with dl_cols[1]:
            st.download_button("📅 Export ICS (calendar)", ics_str, "itinerary.ics",
                               mime="text/calendar", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
#  FOOTER
# =========================================================
st.markdown(
    "<div class='small'>Data: CSV TripAdvisor (usage pédagogique). Carto: pydeck/Deck.gl. "
    "Scoring interactif (Bayesian + popularité) et itinéraire optimisé 2-opt.</div>",
    unsafe_allow_html=True,
)
