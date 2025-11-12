import os, sys, boto3, json, gzip, io
from math import radians, sin, cos, asin, sqrt
from flask import Flask, render_template, request, jsonify
from botocore.exceptions import ClientError

USE_SM = os.getenv("USE_SM", "false").lower() == "true"
SM_ENDPOINT = os.getenv("SM_ENDPOINT", "")
sm_rt = boto3.client("sagemaker-runtime") if USE_SM else None
RAW_BUCKET = os.getenv("RAW_BUCKET")  # e.g. traffic-ai-raw-<acct>-us-east-1
CARPARK_PREFIX = "raw/lta/carparks/"

# Normalise key names before loading
if os.getenv("GOOGLE_MAPS_API_KEY") and not os.getenv("GOOGLE_MAP_KEY"):
    os.environ["GOOGLE_MAP_KEY"] = os.getenv("GOOGLE_MAPS_API_KEY")

if not os.getenv("GOOGLE_MAP_KEY"):
    try:
        region = os.getenv("AWS_REGION", "us-east-1")
        google_secret_id = os.getenv("GOOGLE_MAPS_SECRET_ID", "traffic-ai/google-map-key")
        sm = boto3.client("secretsmanager", region_name=region)
        blob = sm.get_secret_value(SecretId=google_secret_id)["SecretString"]
        os.environ["GOOGLE_MAP_KEY"] = json.loads(blob).get("key", "")
    except Exception as e:
        print(f"[startup] Could not load Google Maps key: {e}")



def _s3_client():
    return boto3.client("s3", region_name="us-east-1")  # your region

def _get_latest_key_for_prefix(s3, bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objs = resp.get("Contents", [])
    if not objs: return None
    latest = max(objs, key=lambda o: o["LastModified"])
    return latest["Key"]

def _load_json_from_s3(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

# --- make src importable (backend lives under src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Unified imports (no 'src.' prefix anywhere)
from backend.data.lta_client import client as lta_client
from backend.data.weather_client import get_weather_data
from backend.data.google_direction import route_google

# Prediction modules (guarded)
try:
    from backend.prediction.route_predictor import get_route_predictor
    from backend.models.model_loader import get_model_loader
    PREDICTION_AVAILABLE = True
except Exception as e:
    print(f"Prediction module not available: {e}")
    PREDICTION_AVAILABLE = False

# Initialize models at import (optional; can defer until first request)
def initialize_models():
    if not PREDICTION_AVAILABLE:
        print("Prediction modules not available - skipping model initialization")
        return False
    try:
        print("Initializing traffic prediction models...")
        model_loader = get_model_loader()
        models_loaded = model_loader.load_models()
        cnt = len(models_loaded) if models_loaded else 0
        print(f"Model initialization completed! {cnt}/3 models loaded" if cnt else "No models loaded")
        return cnt > 0
    except Exception:
        import traceback; traceback.print_exc()
        return False

models_initialized = False if USE_SM else (initialize_models() if PREDICTION_AVAILABLE else False)

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static",
)

# ---------- health ----------
@app.get("/health")
def health():
    models_loaded_count = 0
    if PREDICTION_AVAILABLE:
        try:
            model_loader = get_model_loader()
            models_loaded_count = len(model_loader.models) if model_loader.loaded else 0
        except:
            pass
    return jsonify(
        ok=True,
        models_loaded=models_initialized,
        models_loaded_count=models_loaded_count,
        prediction_available=PREDICTION_AVAILABLE,
        message=f"{models_loaded_count}/3 AI models loaded successfully" if models_initialized else "AI models not available",
    ), 200

@app.get("/healthz")
def healthz():
    return health()

# ---------- basic pages ----------
@app.route("/")
def home():
    return render_template("index.html", title="Local")

@app.route("/get_weather", methods=["GET"])
def get_weather():
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
        "resolved_area": data.get("resolved_area") or (city or "Singapore"),
    }
    return jsonify(safe)

# ---------- routing (Google Directions) ----------
@app.post("/route")
def route():
    body = request.get_json(force=True) or {}
    origin = body.get("origin")
    destination = body.get("destination")
    filters = body.get("filters", {})
    waypoints = body.get("stops")  # optional

    def validate_location(loc, name):
        if isinstance(loc, str):
            if not loc.strip():
                return None, f"{name} address cannot be empty"
            return loc.strip(), None
        elif isinstance(loc, list):
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

    origin_processed, origin_err = validate_location(origin, "origin")
    if origin_err:
        return jsonify(error=origin_err), 400

    dest_processed, dest_err = validate_location(destination, "destination")
    if dest_err:
        return jsonify(error=dest_err), 400

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
    body = request.get_json(force=True) or {}
    origin = body.get("origin")
    destination = body.get("destination")

    # Validate inputs (keeps same shape as your original)
    if not (isinstance(origin, list) and isinstance(destination, list)
            and len(origin) == 2 and len(destination) == 2):
        return jsonify(error="origin/destination must be [lat, lon]"), 400

    # ---- SageMaker path ----
    if USE_SM:
        try:
            resp = sm_rt.invoke_endpoint(
                EndpointName=SM_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(body).encode("utf-8"),
            )
            payload = resp["Body"].read()
            data = json.loads(payload)
            # Add a marker so you can prove it's SageMaker:
            return jsonify(data), 200
        except Exception as e:
            return jsonify(error=f"SageMaker invoke failed: {e}"), 502

    # ---- Local path (only if not using SM) ----
    if not PREDICTION_AVAILABLE:
        return jsonify(error="Prediction modules are not available"), 503
    if not models_initialized:
        return jsonify(error="AI models are not loaded. Please check server logs."), 503

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

@app.get("/speedbands")
def speedbands():
    import re
    hour = request.args.get("hour", default=0, type=int)

    def clean_name(name):
        # remove things in parentheses like (PIE)
        return re.sub(r"\s*\(.*?\)\s*", "", name).strip()

    def band_for(base, hour):
        b = base + (hour % 4) - 2
        return max(1, min(8, b))

    roads = [
        ("Pan Island Expressway (PIE)", 1.3500, 103.7800, 1.3500, 103.9600, 5),
        ("Ayer Rajah Expressway (AYE)", 1.2800, 103.7500, 1.2800, 103.8800, 4),
        ("East Coast Parkway (ECP)", 1.3000, 103.8600, 1.3000, 103.9800, 6),
        ("Central Expressway (CTE)", 1.3200, 103.8500, 1.3800, 103.8500, 3),
        ("Orchard Road", 1.3040, 103.8250, 1.3040, 103.8400, 2),
    ]

    rows = []
    for name, slat, slon, elat, elon, base in roads:
        clean = clean_name(name)
        rows.append({
            "LinkID": clean.replace(" ", "_"),
            "RoadName": clean,
            "RoadCategory": 3,
            "SpeedBand": band_for(base, hour),
            "MinimumSpeed": 40,
            "MaximumSpeed": 80,
            "StartLat": slat,
            "StartLon": slon,
            "EndLat": elat,
            "EndLon": elon,
        })
    return jsonify(rows), 200


@app.get("/carparks")
def carparks_nearby():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    radius_km = request.args.get("radius_km", default=1.0, type=float)
    limit = request.args.get("limit", default=150, type=int)

    rows = None

    # 1) Try S3 snapshot first (preferred)
    if RAW_BUCKET:
        try:
            s3 = _s3_client()
            key = _get_latest_key_for_prefix(s3, RAW_BUCKET, CARPARK_PREFIX)
            if key:
                snap = _load_json_from_s3(s3, RAW_BUCKET, key)
                # Expect same structure your LTA client returns: {"value": [...]}
                rows = (snap.get("value") or snap.get("data", {}).get("value") or [])
        except ClientError as e:
            # fall back silently
            print(f"[carparks] S3 read failed: {e}")

    # 2) Fallback to live LTA if S3 not available
    if rows is None:
        c = lta_client()
        rows, skip = [], 0
        PAGE, MAXP = 500, 20
        while True:
            page = c.carpark_availability(skip=skip).get("value", [])
            rows.extend(page)
            if len(page) < PAGE or len(rows) >= PAGE * MAXP:
                break
            skip += len(page)

    # Parse into your stable shape
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

    if lat is not None and lon is not None:
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

    def lots_val(v):
        try:
            return int(v)
        except Exception:
            return -1

    parsed.sort(key=lambda x: (-lots_val(x["AvailableLots"]), x["Development"] or ""))

    return jsonify({"count": len(parsed), "carparks": parsed[:limit]})

# ---------- run (local dev) ----------
if __name__ == "__main__":
    print("Server starting...")
    print(f"Models initialized: {models_initialized}")
    print(f"Prediction available: {PREDICTION_AVAILABLE}")
    app.run(host="0.0.0.0", port=8000, debug=True)
