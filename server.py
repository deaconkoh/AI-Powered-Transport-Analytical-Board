import os, sys
from flask import Flask, render_template, request, jsonify
from math import radians, sin, cos, asin, sqrt
from src.backend.data.lta_client import client as lta_client

# --- make src importable (backend lives under src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- app-specific imports
from backend.data.weather_client import get_weather_data
from backend.data.google_direction import route_google 

# Import prediction modules with error handling
try:
    from backend.prediction.route_predictor import get_route_predictor
    from backend.models.model_loader import get_model_loader
    PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"Prediction module not available: {e}")
    PREDICTION_AVAILABLE = False
except Exception as e:
    print(f"Error importing prediction module: {e}")
    PREDICTION_AVAILABLE = False

# Initialize models when server starts
def initialize_models():
    """Initialize AI models on server startup"""
    if not PREDICTION_AVAILABLE:
        print("Prediction modules not available - skipping model initialization")
        return False
        
    try:
        print("Initializing traffic prediction models...")
        model_loader = get_model_loader()
        models_loaded = model_loader.load_models()
        
        # Check how many models actually loaded
        models_loaded_count = len(models_loaded) if models_loaded else 0
        
        if models_loaded_count > 0:
            print(f"Model initialization completed! {models_loaded_count}/3 models loaded")
            return True
        else:
            print("Model initialization failed - no models loaded")
            return False
            
    except Exception as e:
        print(f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Call initialization when module loads
models_initialized = initialize_models() if PREDICTION_AVAILABLE else False

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static",
)

# ---------- basic pages ----------
@app.route("/")
def home():
    return render_template("index.html", title="Local")

@app.route("/get_weather", methods=["GET"])
def get_weather():
    # Accept either ?city=... or ?lat=...&lon=...
    city = request.args.get("city")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    data = get_weather_data(city=city, lat=lat, lon=lon)

    safe = {
        "forecast": data.get("forecast") or "Fair",
        "temperature": data.get("temperature") if data.get("temperature") is not None else 30,
        "humidity": data.get("humidity") if data.get("humidity") is not None else 75,
        "wind_speed": data.get("wind_speed") if data.get("wind_speed") is not None else 15,
        "feels_like": data.get("feels_like") if data.get("feels_like") is not None else 32,
        # helpful for the UI label
        "resolved_area": data.get("resolved_area") or (city or "Singapore"),
    }
    return jsonify(safe)

# ---------- health ----------
@app.get("/health")
def health():
    models_loaded_count = 0
    if PREDICTION_AVAILABLE:
        try:
            model_loader = get_model_loader()
            models_loaded_count = len(model_loader.models) if model_loader.loaded else 0
        except:
            models_loaded_count = 0
    
    return jsonify(
        ok=True, 
        models_loaded=models_initialized,
        models_loaded_count=models_loaded_count,
        prediction_available=PREDICTION_AVAILABLE,
        message=f"{models_loaded_count}/3 AI models loaded successfully" if models_initialized else "AI models not available"
    ), 200

# ---------- routing (Google Directions) ----------
@app.post("/route")
def route():
    body = request.get_json(force=True) or {}
    origin = body.get("origin")
    destination = body.get("destination")
    filters = body.get("filters", {})
    waypoints = body.get("stops")  # optional

    def validate_location(loc, name):
        """
        Validate location input - accepts either:
        - String (address): "Changi Airport, Singapore"
        - List of 2 numbers (coordinates): [1.281, 103.863]
        
        Returns: (processed_value, error_message)
        """
        if isinstance(loc, str):
            # It's an address string
            if not loc.strip():
                return None, f"{name} address cannot be empty"
            return loc.strip(), None
        
        elif isinstance(loc, list):
            # It's coordinates
            if len(loc) != 2:
                return None, f"{name} must be [lat, lng] array"
            
            lat, lng = loc
            if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
                return None, f"{name} coordinates must be numbers"
            
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                return None, f"{name} has invalid lat/lng values"
            
            return tuple(loc), None
        
        else:
            return None, f"{name} must be an address string or [lat, lng] array"

    # Validate origin
    origin_processed, origin_err = validate_location(origin, "origin")
    if origin_err:
        return jsonify(error=origin_err), 400

    # Validate destination
    dest_processed, dest_err = validate_location(destination, "destination")
    if dest_err:
        return jsonify(error=dest_err), 400

    # Validate waypoints if provided
    waypoints_processed = None
    if waypoints:
        if not isinstance(waypoints, list):
            return jsonify(error="waypoints must be an array"), 400
        
        waypoints_processed = []
        for i, wp in enumerate(waypoints):
            wp_processed, wp_err = validate_location(wp, f"waypoint[{i}]")
            if wp_err:
                return jsonify(error=wp_err), 400
            waypoints_processed.append(wp_processed)

    # Call provider
    result, err = route_google(
        origin_processed,
        dest_processed,
        filters,
        waypoints=waypoints_processed
    )
    
    if err:
        msg, code = err
        return jsonify(error=msg), code
    
    return jsonify(result), 200

# ---------- AI Prediction Route ----------
@app.post("/predict_route")
def predict_route():
    """Get AI-powered route predictions"""
    if not PREDICTION_AVAILABLE:
        return jsonify(error="Prediction modules are not available"), 503
        
    if not models_initialized:
        return jsonify(error="AI models are not loaded. Please check server logs."), 503
        
    body = request.get_json(force=True) or {}
    origin = body.get("origin")
    destination = body.get("destination")
    
    if not (isinstance(origin, list) and isinstance(destination, list) and len(origin) == 2 and len(destination) == 2):
        return jsonify(error="origin/destination must be [lat, lon]"), 400
    
    try:
        predictor = get_route_predictor()
        prediction = predictor.predict_route_conditions(tuple(origin), tuple(destination))
        
        if "error" in prediction:
            return jsonify(prediction), 500
            
        return jsonify(prediction), 200
        
    except Exception as e:
        return jsonify(error=f"Prediction failed: {str(e)}"), 500

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi, dlmb = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlmb/2)**2
    return 2 * R * asin(sqrt(a))

@app.get("/carparks")
def carparks_nearby():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    radius_km = request.args.get("radius_km", default=1.0, type=float)
    limit = request.args.get("limit", default=150, type=int)  # cap for UI

    c = lta_client()

    # Page through DataMall (use your existing paging util)
    rows, skip = [], 0
    PAGE, MAXP = 500, 20
    while True:
        page = c.carpark_availability(skip=skip).get("value", [])
        rows.extend(page)
        if len(page) < PAGE or len(rows) >= PAGE * MAXP:
            break
        skip += len(page)

    # Parse & normalize results
    parsed = []
    for cp in rows:
        loc = (cp.get("Location") or "").strip()  # "lat lon"
        try:
            slat, slon = [float(x) for x in loc.split()]
        except Exception:
            continue
        parsed.append({
            "Development": cp.get("Development"),
            "Location": {"Latitude": slat, "Longitude": slon},
            "AvailableLots": cp.get("AvailableLots"),
            "LotType": cp.get("LotType"),
            "Agency": cp.get("Agency"),
            "Area": cp.get("Area"),
        })

    # Mode 1: coords provided → filter by radius
    if lat is not None and lon is not None:
        from math import radians, sin, cos, asin, sqrt
        def haversine_km(a_lat, a_lon, b_lat, b_lon):
            R = 6371.0
            p1, p2 = radians(a_lat), radians(b_lat)
            dphi, dlmb = radians(b_lat - a_lat), radians(b_lon - a_lon)
            x = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlmb/2)**2
            return 2 * R * asin(sqrt(x))

        nearby = []
        for cp in parsed:
            d = haversine_km(lat, lon, cp["Location"]["Latitude"], cp["Location"]["Longitude"])
            if d <= radius_km:
                cp["DistanceKm"] = round(d, 3)
                nearby.append(cp)

        nearby.sort(key=lambda x: x["DistanceKm"])
        return jsonify({"count": len(nearby), "carparks": nearby[:limit]})

    # Mode 2: no coords → return all (sorted by AvailableLots desc, then name)
    def lots_val(v):
        try:
            return int(v)
        except Exception:
            return -1

    parsed.sort(key=lambda x: (-lots_val(x["AvailableLots"]), x["Development"] or ""))

    return jsonify({"count": len(parsed), "carparks": parsed[:limit]})

# ---------- run ----------
if __name__ == "__main__":
    print(f"Server starting...")
    print(f"Models initialized: {models_initialized}")
    print(f"Prediction available: {PREDICTION_AVAILABLE}")
    # port 8000 to match your earlier setup
    app.run(host="0.0.0.0", port=8000, debug=True)