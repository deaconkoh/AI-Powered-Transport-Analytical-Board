# weather_client.py
import os
import math
import requests
from dotenv import load_dotenv
from statistics import mean
from typing import Optional, Tuple

load_dotenv()

API_KEY = os.getenv("DATA_GOV_SG_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
BASE = "https://api.data.gov.sg/v1/environment"

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _two_hour_forecast_raw():
    url = f"{BASE}/2-hour-weather-forecast"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def _find_area_centroid(area_json, city: str) -> Optional[Tuple[str, float, float]]:
    """
    Find an area by name (case-insensitive, substring ok) and return (area_name, lat, lon).
    """
    items = area_json.get("area_metadata", [])
    if not city:
        return None
    city_l = city.strip().lower()

    # First try exact (case-insensitive), then substring
    exact = [a for a in items if a.get("name","").lower() == city_l]
    if exact:
        a = exact[0]
        loc = a.get("label_location", {}) or {}
        return (a.get("name"), loc.get("latitude"), loc.get("longitude"))

    subs = [a for a in items if city_l in a.get("name","").lower()]
    if subs:
        a = subs[0]
        loc = a.get("label_location", {}) or {}
        return (a.get("name"), loc.get("latitude"), loc.get("longitude"))

    return None

def _area_mode_forecast(area_json) -> Optional[str]:
    items = (area_json.get("items") or [])
    if not items:
        return None
    forecasts = items[0].get("forecasts") or []
    if not forecasts:
        return None
    counts = {}
    for f in forecasts:
        label = (f.get("forecast") or "").strip()
        if label:
            counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get) if counts else None

def _area_specific_forecast(area_json, target_area_name: str) -> Optional[str]:
    items = (area_json.get("items") or [])
    if not items:
        return None
    forecasts = items[0].get("forecasts") or []
    target_l = (target_area_name or "").lower()
    for f in forecasts:
        if (f.get("area","").lower() == target_l) and f.get("forecast"):
            return f["forecast"].strip()
    return None

def _nearest_station_value(env_endpoint: str, value_key: str, target_lat: float, target_lon: float) -> Optional[float]:
    """
    From an environment endpoint (e.g. air-temperature), pick the latest reading
    from the station nearest to (target_lat, target_lon).
    """
    url = f"{BASE}/{env_endpoint}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json()

    stations = (j.get("metadata", {}) or {}).get("stations", []) or []
    items = j.get("items", []) or []
    if not stations or not items:
        return None

    latest = items[0]
    readings = latest.get("readings", []) or []
    if not readings:
        return None

    # Map station_id -> (lat, lon)
    coords = {}
    for s in stations:
        sid = s.get("id")
        loc = s.get("location", {}) or {}
        if sid and "latitude" in loc and "longitude" in loc:
            coords[sid] = (loc["latitude"], loc["longitude"])

    # Choose reading whose station is nearest to the target point
    best_val, best_dist = None, float("inf")
    for rd in readings:
        sid = rd.get("station_id")
        val = rd.get(value_key)
        if sid in coords and isinstance(val, (int, float)):
            lat, lon = coords[sid]
            d = _haversine_km(target_lat, target_lon, lat, lon)
            if d < best_dist:
                best_dist = d
                best_val = val

    return round(best_val, 1) if isinstance(best_val, (int, float)) else None

def _avg_latest_reading(env_endpoint: str, value_key: str) -> Optional[float]:
    """Average across all stations (fallback when no city provided/found)."""
    url = f"{BASE}/{env_endpoint}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json()
    items = j.get("items", []) or []
    if not items:
        return None
    readings = items[0].get("readings", []) or []
    vals = [rd.get(value_key) for rd in readings if isinstance(rd.get(value_key), (int, float))]
    return round(mean(vals), 1) if vals else None

def _heat_index_c(temp_c: Optional[float], rh_pct: Optional[float]) -> Optional[int]:
    if temp_c is None or rh_pct is None:
        return None
    T = temp_c * 9/5 + 32
    R = rh_pct
    HI = (-42.379 + 2.04901523*T + 10.14333127*R
          - 0.22475541*T*R - 0.00683783*T*T - 0.05481717*R*R
          + 0.00122874*T*T*R + 0.00085282*T*R*R - 0.00000199*T*T*R*R)
    return int(round((HI - 32) * 5/9))

# weather_client.py (only showing new/changed bits)
def _nearest_area_name(area_json, lat, lon) -> Optional[str]:
    """Pick the nearest 2-hr-forecast area to (lat, lon)."""
    metas = area_json.get("area_metadata", []) or []
    best_name, best_d = None, float("inf")
    for m in metas:
        loc = m.get("label_location", {}) or {}
        alat, alon = loc.get("latitude"), loc.get("longitude")
        if isinstance(alat, (int,float)) and isinstance(alon, (int,float)):
            d = _haversine_km(lat, lon, alat, alon)
            if d < best_d:
                best_d, best_name = d, m.get("name")
    return best_name

def get_weather_data(city: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    """
    If lat/lon provided: use nearest stations to those coords and nearest-area forecast.
    Else if city provided: use centroid of that area.
    Else: island-wide aggregates & mode forecast.
    """
    two_hr = _two_hour_forecast_raw()

    target_lat = target_lon = None
    resolved_area = None

    if isinstance(lat, (int,float)) and isinstance(lon, (int,float)):
        target_lat, target_lon = float(lat), float(lon)
        resolved_area = _nearest_area_name(two_hr, target_lat, target_lon)
        forecast = _area_specific_forecast(two_hr, resolved_area) or _area_mode_forecast(two_hr)
    else:
        target = _find_area_centroid(two_hr, city) if city else None
        if target:
            resolved_area, target_lat, target_lon = target
            forecast = _area_specific_forecast(two_hr, resolved_area) or _area_mode_forecast(two_hr)
        else:
            forecast = _area_mode_forecast(two_hr)

    if target_lat is not None and target_lon is not None:
        temp = _nearest_station_value("air-temperature", "value", target_lat, target_lon)
        humidity = _nearest_station_value("relative-humidity", "value", target_lat, target_lon)
        wind_ms = _nearest_station_value("wind-speed", "value", target_lat, target_lon)
    else:
        temp = _avg_latest_reading("air-temperature", "value")
        humidity = _avg_latest_reading("relative-humidity", "value")
        wind_ms = _avg_latest_reading("wind-speed", "value")

    wind_kmh = round(wind_ms * 3.6, 1) if isinstance(wind_ms, (int, float)) else None
    feels_like = _heat_index_c(temp, humidity)

    return {
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind_kmh,
        "forecast": forecast,
        "feels_like": feels_like,
        "resolved_area": resolved_area,
    }


if __name__ == "__main__":
    from pprint import pprint
    print("No city:")
    pprint(get_weather_data())
    print("\nCity=Punggol:")
    pprint(get_weather_data(city="Punggol"))
