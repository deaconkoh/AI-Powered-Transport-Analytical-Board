import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_SG_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

BASE = "https://api.data.gov.sg/v1/environment"

def get_weather_data():
    """
    Returns a small bundle of public weather info for the UI.
    You can expand this later (forecast, PSI, etc.).
    """
    # air temperature (works w/o key)
    air_url = f"{BASE}/air-temperature"
    # rainfall (works w/o key)
    rain_url = f"{BASE}/rainfall"

    out = {}

    # air temperature
    r1 = requests.get(air_url, headers=HEADERS, timeout=15)
    r1.raise_for_status()
    air = r1.json()
    out["air_temperature"] = air

    # rainfall
    r2 = requests.get(rain_url, headers=HEADERS, timeout=15)
    r2.raise_for_status()
    rain = r2.json()
    out["rainfall"] = rain

    return out

if __name__ == "__main__":
    # optional: quick manual test
    from pprint import pprint
    pprint(get_weather_data())
