import os, requests
from typing import Optional, Iterable, Tuple, Dict, Any, Union
from dotenv import load_dotenv
load_dotenv()

def geocode_address(
    address: str,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = 10.0,
) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    """
    Convert an address string to (lat, lng) coordinates using Google Geocoding API.
    Returns: ((lat, lng), None) on success, or (None, error_message) on failure
    """
    key = api_key or os.getenv("GOOGLE_MAP_KEY")
    if not key:
        return None, "GOOGLE_MAP_KEY env not set"
    
    s = session or requests.Session()
    try:
        r = s.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": address,
                "key": key,
                "region": "sg"  # Bias results to Singapore
            },
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return None, f"Geocoding network error: {e}"
    
    status = data.get("status")
    if status != "OK":
        return None, f"Geocoding failed: {status}"
    
    results = data.get("results", [])
    if not results:
        return None, "No geocoding results found"
    
    location = results[0]["geometry"]["location"]
    return (location["lat"], location["lng"]), None


def route_google(
    origin: Union[Tuple[float, float], str],
    dest: Union[Tuple[float, float], str],
    filters: Dict[str, Any],
    waypoints: Optional[Iterable[Union[Tuple[float, float], str]]] = None,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: float = 15.0,
):
    key = api_key or os.getenv("GOOGLE_MAP_KEY")
    if not key:
        return None, ("GOOGLE_MAP_KEY env not set", 500)

    def validate_coords(pt):
        lat, lng = pt
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        return {"latitude": lat, "longitude": lng}

    def process_location(loc):
        """Handle both address strings and (lat, lng) tuples"""
        if isinstance(loc, str):
            # It's an address, geocode it
            coords, error = geocode_address(loc, api_key=key, session=session)
            if error:
                return None, error
            return validate_coords(coords), None
        else:
            # It's already coordinates
            result = validate_coords(loc)
            return result, None if result else "Invalid lat/lng"

    o, o_err = process_location(origin)
    if o_err:
        return None, (f"Origin error: {o_err}", 400)
    
    d, d_err = process_location(dest)
    if d_err:
        return None, (f"Destination error: {d_err}", 400)

    # Build route modifiers
    route_modifiers = {
        "avoidTolls": bool(filters.get("avoidERP")),
        "avoidHighways": bool(filters.get("avoidHighway")),
        "avoidFerries": bool(filters.get("avoidFerries"))
    }

    # Build request body for Routes API v2
    request_body = {
        "origin": {
            "location": {
                "latLng": o
            }
        },
        "destination": {
            "location": {
                "latLng": d
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE" if filters.get("fastest", True) else "TRAFFIC_UNAWARE",
        "computeAlternativeRoutes": False,
        "routeModifiers": route_modifiers,
        "languageCode": "en-US",
        "units": "METRIC"
    }

    # Add waypoints if provided
    if waypoints:
        wps = list(waypoints)
        if len(wps) > 25:  # Routes API v2 allows up to 25 waypoints
            return None, ("Waypoint limit exceeded (max 25)", 400)
        
        intermediates = []
        for i, wp in enumerate(wps):
            if isinstance(wp, str):
                # Geocode address
                wp_coords, wp_err = geocode_address(wp, api_key=key, session=session)
                if wp_err:
                    return None, (f"Waypoint {i+1} geocoding error: {wp_err}", 400)
                wp_coords = validate_coords(wp_coords)
            else:
                # Validate coordinates
                wp_coords = validate_coords(wp)
            
            if not wp_coords:
                return None, (f"Invalid waypoint {i+1} coordinates", 400)
            
            intermediates.append({
                "location": {
                    "latLng": wp_coords
                }
            })
        
        if intermediates:
            request_body["intermediates"] = intermediates
            request_body["optimizeWaypointOrder"] = True

    # Set field mask to specify what data we want back
    field_mask = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs"
    if filters.get("fastest", True):
        field_mask += ",routes.travelAdvisory.speedReadingIntervals"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": field_mask
    }

    s = session or requests.Session()
    try:
        r = s.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            json=request_body,
            headers=headers,
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return None, (f"Network error: {e}", 502)

    # Check for errors in response
    if "error" in data:
        error_msg = data["error"].get("message", "Unknown error")
        error_code = data["error"].get("code", 502)
        return None, (f"Google Routes API error: {error_msg}", error_code)

    routes = data.get("routes", [])
    if not routes:
        return None, ("No route returned", 400)

    route0 = routes[0]
    
    # Extract distance and duration
    dist_m = route0.get("distanceMeters", 0)
    dur_s_str = route0.get("duration", "0s")
    # Parse duration string (e.g., "1234s" -> 1234)
    dur_s = int(dur_s_str.rstrip('s')) if dur_s_str.endswith('s') else 0

    # Extract legs for detailed route information
    legs = route0.get("legs", [])

    result = {
        "distance_km": round(dist_m / 1000.0, 3),
        "eta_seconds": int(dur_s),
        "appliedFilters": {
            "tolls": bool(filters.get("avoidERP")),
            "motorways": bool(filters.get("avoidHighway")),
            "ferries": bool(filters.get("avoidFerries")),
            "strategy": "Fastest" if filters.get("fastest", True) else "Shortest",
        },
        "overview_polyline": route0.get("polyline", {}).get("encodedPolyline"),
        "legs": legs,
        "provider": "google_routes_v2",
    }
    return result, None