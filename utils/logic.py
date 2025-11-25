"""
Logique métier : scoring bayésien, sélection des restaurants et génération d'itinéraires.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PRICE_BUCKETS = {
    "Any": (None, None),
    "Economy": (None, 1.5),
    "Moderate": (1.5, 3.0),
    "Premium": (3.0, None),
}

PRICE_TO_EUROS = {
    1: 20,
    2: 40,
    3: 75,
    4: 120,
}


def compute_scores(df: pd.DataFrame, bayes_weight: int, rating_weight: float) -> pd.DataFrame:
    """Ajoute les colonnes de score pondéré et pénalise les pièges à touristes."""
    df = df.copy()
    if "rating" not in df.columns:
        df["score_final"] = 0.0
        return df

    m = max(1, bayes_weight)
    C = df["rating"].dropna().mean()
    v = df["num_reviews"].fillna(0)
    R = df["rating"].fillna(C)
    df["rating_weighted"] = (v * R + m * C) / (v + m)

    mu = df.groupby("city")["rating"].transform("mean")
    sd = df.groupby("city")["rating"].transform("std").replace(0, np.nan)
    z = (df["rating"] - mu) / sd
    z = z.clip(-2, 2).fillna(0.0)

    scaler = MinMaxScaler()
    df["hotness_scaled"] = scaler.fit_transform(np.asarray(z, dtype=float).reshape(-1, 1))

    rating_weight = float(np.clip(rating_weight, 0.0, 1.0))
    popularity_weight = 1.0 - rating_weight
    df["score_final"] = rating_weight * df["rating_weighted"] + popularity_weight * df["hotness_scaled"]

    mask_tourist = (df["num_reviews"] > 1000) & (df["rating"] < 4.0)
    df.loc[mask_tourist, "score_final"] *= 0.8
    return df


def _filter_by_preferences(
    df: pd.DataFrame,
    cuisines: Sequence[str],
    price_level: str,
) -> pd.DataFrame:
    data = df.copy()
    if cuisines:
        data = data[data["cuisine_list"].apply(lambda L: any(c in L for c in cuisines))]
    low, high = PRICE_BUCKETS.get(price_level, (None, None))
    if low is not None:
        data = data[data["price_num"] > low]
    if high is not None:
        data = data[data["price_num"] <= high]
    return data


def _pick_meal(
    candidates: pd.DataFrame,
    used_restos: set,
    last_cuisine: Optional[str],
) -> Tuple[Optional[pd.Series], Optional[str]]:
    for _, row in candidates.iterrows():
        cuisine_major = row.get("cuisine_major")
        if last_cuisine and cuisine_major == last_cuisine:
            continue
        if row["restaurant_name"] in used_restos:
            continue
        return row, cuisine_major or last_cuisine
    return None, last_cuisine


def _estimate_cost(price_num: Optional[float]) -> float:
    if price_num is None or np.isnan(price_num):
        return 40.0
    bucket = int(round(price_num))
    return float(PRICE_TO_EUROS.get(bucket, 40.0))


def build_itinerary(
    df: pd.DataFrame,
    ordered_cities: Sequence[Dict],
    days: int,
    cuisines: Sequence[str],
    price_level: str,
    start_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Construit l'itinéraire et retourne (tableau, points carte, budget estimé)."""
    if not ordered_cities or days <= 0:
        return (
            pd.DataFrame(columns=["date", "city", "country", "lunch_name", "dinner_name"]),
            pd.DataFrame(columns=["restaurant_name", "latitude", "longitude"]),
            0.0,
        )

    sequence = list(ordered_cities)
    while len(sequence) < days:
        sequence.extend(ordered_cities)
    sequence = sequence[:days]

    rows = []
    restos = []
    used_restos: set = set()
    last_cuisine = None
    estimated_budget = 0.0

    for day_idx, stop in enumerate(sequence):
        sub = df[(df["city"] == stop["city"]) & (df["country"] == stop["country"])]
        filtered = _filter_by_preferences(sub, cuisines, price_level).sort_values("score_final", ascending=False)
        fallback_msg = ""
        if filtered.empty:
            fallback_msg = "Aucun restaurant correspondant — élargis les filtres."
            filtered = sub.sort_values("score_final", ascending=False)

        lunch, last_cuisine = _pick_meal(filtered, used_restos, last_cuisine)
        dinner, last_cuisine = _pick_meal(filtered, used_restos, last_cuisine)

        lunch_name = lunch["restaurant_name"] if lunch is not None else fallback_msg
        dinner_name = dinner["restaurant_name"] if dinner is not None else fallback_msg

        event_date = start_date + timedelta(days=day_idx)
        rows.append(
            {
                "date": event_date,
                "city": stop["city"],
                "country": stop["country"],
                "lunch_name": lunch_name,
                "dinner_name": dinner_name,
            }
        )

        for meal in [lunch, dinner]:
            if meal is not None:
                used_restos.add(meal["restaurant_name"])
                restos.append(
                    {
                        "restaurant_name": meal["restaurant_name"],
                        "city": stop["city"],
                        "country": stop["country"],
                        "latitude": meal["latitude"],
                        "longitude": meal["longitude"],
                        "rating": meal.get("rating", np.nan),
                        "price_level": meal.get("price_level", ""),
                        "cuisine": meal.get("cuisine", ""),
                    }
                )
                estimated_budget += _estimate_cost(meal.get("price_num"))

    return pd.DataFrame(rows), pd.DataFrame(restos), estimated_budget


def make_ics(itinerary_df: pd.DataFrame) -> str:
    """Génère un export ICS basé sur les vraies dates."""
    ics = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CulinaryRoadTrip//FR"]
    for _, row in itinerary_df.iterrows():
        row_date: date = row["date"]
        for when, title, field in [("12:30", "Lunch", "lunch_name"), ("19:30", "Dinner", "dinner_name")]:
            h, m = map(int, when.split(":"))
            dtstart = datetime.combine(row_date, datetime.min.time()) + timedelta(hours=h, minutes=m)
            dtend = dtstart + timedelta(hours=1, minutes=30)
            summary = f"{title} — {row['city']} — {row[field]}"
            ics += [
                "BEGIN:VEVENT",
                f"DTSTART:{dtstart.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{dtend.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:{summary}",
                "END:VEVENT",
            ]
    ics.append("END:VCALENDAR")
    return "\n".join(ics)
