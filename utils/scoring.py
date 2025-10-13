import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_scores(df):
    """Ajoute rating_weighted, hotness_local, score_final."""
    m = 50
    C = df['rating'].mean()
    v = df['num_reviews'].fillna(0)
    R = df['rating'].fillna(C)
    df['rating_weighted'] = (v * R + m * C) / (v + m)

    mu = df.groupby('city')['rating'].transform('mean')
    sd = df.groupby('city')['rating'].transform('std').replace(0,np.nan)
    z = ((df['rating'] - mu) / sd).clip(-2,2).fillna(0)
    scaler = MinMaxScaler()
    df['hotness_scaled'] = scaler.fit_transform(z.values.reshape(-1,1))
    df['score_final'] = 0.7 * df['rating_weighted'] + 0.3 * df['hotness_scaled']
    return df