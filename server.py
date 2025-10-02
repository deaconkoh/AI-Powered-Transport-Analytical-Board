import os, sys
from flask import Flask, render_template, request, jsonify

# --- make src importable (backend lives under src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- app-specific imports
from backend.data.weather_client import get_weather_data
from backend.data.google_direction import route_google 

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
    city = request.form.get("city", "Punggol Coast")
    data = get_weather_data()  # (plug city into your logic later)
    return jsonify(data)

# ---------- health ----------
@app.get("/health")
def health():
    return jsonify(ok=True), 200

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

# ---------- run ----------
if __name__ == "__main__":
    # port 8000 to match your earlier setup
    app.run(host="0.0.0.0", port=8000, debug=True)
