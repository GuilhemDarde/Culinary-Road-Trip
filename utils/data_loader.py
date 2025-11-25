import json
from pathlib import Path

import numpy as np
import pandas as pd

def load_data(csv_path: str):
    """Charge et nettoie le dataset TripAdvisor."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Harmonisation des noms de colonnes
    colmap = {
        'name':'restaurant_name',
        'Restaurant Name':'restaurant_name',
        'lat':'latitude', 'lng':'longitude', 'lon':'longitude',
        'City':'city', 'Country':'country',
        'Rating':'rating', 'Reviews':'num_reviews',
        'Cuisine':'cuisine', 'Cuisines':'cuisine', 'cuisines':'cuisine',
        'Price':'price_level'
    }
    for k,v in colmap.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)

    keep = [c for c in [
        'restaurant_name','address','city','country','latitude','longitude',
        'rating','num_reviews','price_level','cuisine','awards','ranking_position',
        'ranking_category','url'
    ] if c in df.columns]
    df = df[keep].copy()

    # Convertis types
    for col in ['rating','num_reviews','latitude','longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Price numérique
    def price_to_num(x):
        if pd.isna(x): return np.nan
        s = str(x)
        if all(ch == '$' for ch in s): return float(len(s))
        if '-' in s:
            nums = [len(p.strip().replace('€','').replace('$','')) for p in s.split('-')]
            return np.mean(nums)
        lut = {'low':1,'cheap':1,'moderate':2,'mid':2,'medium':2,
               'expensive':3,'premium':3,'fine':4,'luxury':4}
        s2 = s.lower().replace('€','').replace('$','')
        return lut.get(s2,np.nan)
    df['price_num'] = df['price_level'].apply(price_to_num)

    # Cuisine principale
    def parse_cuisine(value):
        if pd.isna(value):
            return tuple()
        parts = [x.strip() for x in str(value).split(',') if x.strip()]
        return tuple(parts)

    if 'cuisine' in df.columns:
        df['cuisine_list'] = df['cuisine'].apply(parse_cuisine)
    else:
        df['cuisine_list'] = [tuple() for _ in range(len(df))]
    df['cuisine_major'] = df['cuisine_list'].apply(lambda L: L[0] if len(L)>0 else np.nan)

    # Centroides villes
    centroids = df.groupby(['country','city'])[['latitude','longitude']].mean().reset_index()

    return df, centroids

def resolve_tripadvisor_csv_path(filename: str = "tripadvisor_european_restaurants.csv") -> Path:
    """
    Retourne un chemin existant vers le CSV TripAdvisor.
    Priorité : data_path.json -> racine projet -> parent direct.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    config_path = repo_root / "data_path.json"
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as fh:
                config = json.load(fh)
            configured = config.get("tripadvisor_csv")
            if configured:
                candidates.append(Path(configured).expanduser())
        except (json.JSONDecodeError, OSError):
            pass

    candidates.extend(
        [
            repo_root / filename,
            repo_root / "european_restaurants.csv",
            repo_root.parent / filename,
        ]
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Impossible de localiser le CSV TripAdvisor. Vérifie data_path.json "
        "ou dépose le fichier dans le dossier du projet."
    )
