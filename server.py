# server.py
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from backend.data.weather_client import get_weather_data
from flask import Flask, render_template, request, jsonify

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static",
)

@app.route("/")
def home():
    return render_template("index.html", title="Local")

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.form.get("city", "Punggol Coast")
    data = get_weather_data()  # plug city into your logic later
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
