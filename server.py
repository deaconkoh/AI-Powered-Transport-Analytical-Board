# server.py
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

@app.route("/get_weather", methods=["POST"])
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
    origin = body.get("origin")        # [lat, lon]
    destination = body.get("destination")
    filters = body.get("filters", {})
    waypoints = body.get("stops")      # optional: [[lat, lon], ...]

    # basic validation
    if not (isinstance(origin, list) and isinstance(destination, list) and len(origin) == 2 and len(destination) == 2):
        return jsonify(error="origin/destination must be [lat, lon]"), 400

    # call provider
    result, err = route_google(tuple(origin), tuple(destination), filters, waypoints=waypoints)
    if err:
        msg, code = err
        return jsonify(error=msg), code
    return jsonify(result), 200

# ---------- run ----------
if __name__ == "__main__":
    # port 8000 to match your earlier setup
    app.run(host="0.0.0.0", port=8000, debug=True)
