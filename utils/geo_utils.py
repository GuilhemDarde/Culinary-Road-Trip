import numpy as np

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def nearest_neighbor_order(points):
    """points = [(city, lat, lon, country)]"""
    if not points: return []
    pts = points.copy()
    route = [pts.pop(0)]
    while pts:
        last = route[-1]
        dists = [(haversine_km(last[1], last[2], p[1], p[2]), i) for i,p in enumerate(pts)]
        _, idx = min(dists, key=lambda x:x[0])
        route.append(pts.pop(idx))
    return route

def distance_for_route(route):
    total = 0
    for i in range(len(route)-1):
        a,b = route[i], route[i+1]
        total += haversine_km(a[1],a[2],b[1],b[2])
    return total