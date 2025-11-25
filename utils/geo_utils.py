"""
Outils géospatiaux : requêtes OSRM, calculs de distances et filtrage spatial.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polyline
import requests

OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving/"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance grand cercle en kilomètres."""
    r = 6371.0
    lat1_rad, lat2_rad = np.radians([lat1, lat2])
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _format_osrm_coordinates(points: Sequence[Tuple[float, float]]) -> str:
    """Format lon,lat pour OSRM."""
    return ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)


def fetch_osrm_route(points: Sequence[Tuple[float, float]]):
    """
    Appelle OSRM pour récupérer distance, durée et géométrie compressée.
    """
    if len(points) < 2:
        raise ValueError("Au moins deux points sont requis pour interroger OSRM.")

    url = OSRM_BASE_URL + _format_osrm_coordinates(points)
    params = {"overview": "full", "geometries": "polyline"}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data.get("routes"):
        raise ValueError("Aucun itinéraire renvoyé par OSRM.")

    route = data["routes"][0]
    decoded = polyline.decode(route["geometry"])
    # OSRM renvoie (lat, lon)
    coords = [(lat, lon) for lat, lon in decoded]
    return {
        "distance_km": route["distance"] / 1000,
        "duration_h": route["duration"] / 3600,
        "polyline": route["geometry"],
        "coordinates": coords,
    }


def filter_cities_along_route(
    city_df: pd.DataFrame,
    route_coords: Sequence[Tuple[float, float]],
    buffer_km: float = 40.0,
) -> pd.DataFrame:
    """
    Retourne les villes dont le centroïde se situe dans un corridor autour de la route.
    """
    if len(route_coords) == 0 or city_df.empty:
        return city_df.iloc[0:0]

    step = max(1, len(route_coords) // 200)
    sampled = route_coords[::step]
    results: List[Dict] = []
    for _, city in city_df.iterrows():
        lat, lon = city["latitude"], city["longitude"]
        distances = [haversine_km(lat, lon, pt[0], pt[1]) for pt in sampled]
        min_dist = min(distances)
        if min_dist <= buffer_km:
            nearest_idx = int(np.argmin(distances))
            results.append(
                {
                    **city.to_dict(),
                    "distance_to_route": float(min_dist),
                    "route_index": nearest_idx,
                }
            )

    if not results:
        return city_df.iloc[0:0]
    return pd.DataFrame(results).sort_values("route_index").reset_index(drop=True)


def get_city_coordinates(city: str, country: str, centroids: pd.DataFrame):
    """
    Renvoie (lat, lon) pour une ville donnée.
    """
    match = centroids[
        (centroids["city"] == city) & (centroids["country"] == country)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return float(row["latitude"]), float(row["longitude"])


def order_cities_along_route(
    city_df: pd.DataFrame,
) -> List[Dict]:
    """
    Retourne une liste ordonnée des villes présentes dans la DataFrame corridor.
    """
    cols = ["city", "country", "latitude", "longitude", "route_index"]
    records = city_df[cols].drop_duplicates(subset=["city", "country"]).to_dict(orient="records")
    return records


def route_position(lat: float, lon: float, route_coords: Sequence[Tuple[float, float]]) -> int:
    """Indice du point de la route le plus proche."""
    if not route_coords:
        return 0
    distances = [haversine_km(lat, lon, pt[0], pt[1]) for pt in route_coords]
    return int(np.argmin(distances))
