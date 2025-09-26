import os, requests
from typing import Optional, Iterable, Tuple, Dict, Any
from dotenv import load_dotenv
load_dotenv()

def route_google(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    filters: Dict[str, Any],
    waypoints: Optional[Iterable[Tuple[float, float]]] = None,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = 15.0,
):
    key = api_key or os.getenv("GOOGLE_MAP_KEY")
    if not key:
        return None, ("GOOGLE_MAP_KEY env not set", 500)

    def norm(pt):
        lat, lng = pt
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        return f"{lat},{lng}"

    o = norm(origin); d = norm(dest)
    if not o or not d:
        return None, ("Invalid lat/lng", 400)

    avoid = []
    if filters.get("avoidERP"):      avoid.append("tolls")
    if filters.get("avoidHighway"):  avoid.append("highways")
    if filters.get("avoidFerries"):  avoid.append("ferries")

    params = {
        "origin": o,
        "destination": d,
        "mode": "driving",
        "units": "metric",
        "key": key,
        "region": "sg",
        "alternatives": "false",
    }

    fastest = filters.get("fastest", True)
    if fastest:
        params["departure_time"] = "now"
        params["traffic_model"] = "best_guess"

    if avoid:
        params["avoid"] = "|".join(avoid)

    if waypoints:
        wps = list(waypoints)
        if len(wps) > 23:
            return None, ("Waypoint limit exceeded (max 23 when optimize:true)", 400)
        wp_strs = [norm(p) for p in wps]
        if any(s is None for s in wp_strs):
            return None, ("Invalid waypoint lat/lng", 400)
        params["waypoints"] = "optimize:true|" + "|".join(wp_strs)

    s = session or requests.Session()
    try:
        r = s.get("https://maps.googleapis.com/maps/api/directions/json",
                  params={k: v for k, v in params.items() if v is not None},
                  timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return None, (f"Network error: {e}", 502)

    status = data.get("status")
    if status != "OK":
        # bubble up Google’s message when available
        emsg = data.get("error_message") or status or "ERROR"
        code = 400 if status in {"ZERO_RESULTS", "INVALID_REQUEST"} else 502
        return None, (f"Google Directions error: {status} - {emsg}", code)

    routes = data.get("routes") or []
    if not routes:
        return None, ("No route returned", 400)

    route0 = routes[0]
    legs = route0.get("legs") or []
    if not legs:
        return None, ("No legs in route", 400)

    dist_m = sum(l["distance"]["value"] for l in legs if "distance" in l)
    # use traffic duration only if we requested it
    if fastest:
        dur_s = sum((l.get("duration_in_traffic") or l.get("duration"))["value"] for l in legs)
    else:
        dur_s = sum(l["duration"]["value"] for l in legs)

    result = {
        "distance_km": round(dist_m / 1000.0, 3),
        "eta_seconds": int(dur_s),
        "appliedFilters": {
            "tolls": bool(filters.get("avoidERP")),
            "motorways": bool(filters.get("avoidHighway")),
            "ferries": bool(filters.get("avoidFerries")),
            "strategy": "Fastest" if fastest else "Shortest",
        },
        "overview_polyline": (route0.get("overview_polyline") or {}).get("points"),
        "legs": legs,
        "provider": "google",
    }
    return result, None
