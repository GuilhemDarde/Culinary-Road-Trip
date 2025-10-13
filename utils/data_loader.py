import pandas as pd
import numpy as np

def load_data(csv_path: str):
    """Charge et nettoie le dataset TripAdvisor."""
    df = pd.read_csv(csv_path)

    # Harmonisation des noms de colonnes
    colmap = {
        'name':'restaurant_name',
        'Restaurant Name':'restaurant_name',
        'lat':'latitude', 'lng':'longitude', 'lon':'longitude',
        'City':'city', 'Country':'country',
        'Rating':'rating', 'Reviews':'num_reviews',
        'Cuisine':'cuisine', 'Price':'price_level'
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
    df['cuisine_list'] = df['cuisine'].fillna('').apply(lambda s: [x.strip() for x in str(s).split(',') if x.strip()])
    df['cuisine_major'] = df['cuisine_list'].apply(lambda L: L[0] if len(L)>0 else np.nan)

    # Centroides villes
    centroids = df.groupby(['country','city'])[['latitude','longitude']].mean().reset_index()

    return df, centroids