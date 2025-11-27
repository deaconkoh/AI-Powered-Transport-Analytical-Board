import json
import os
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
OUTPUT_DATA_FILE = "mock_predictions.json"
OUTPUT_GEO_FILE = "mock_geometry.json"
INPUT_GEOJSON = "roads_wgs84.geojson"
HOURS_TO_PREDICT = 48  # As requested

RAFFLES_MARINA_CITYHALL_BOX = {
    "min_lat": 1.280,
    "max_lat": 1.296,
    "min_lon": 103.845,
    "max_lon": 103.863,
}


DOWNTOWN_CORE_BOX = {
    "min_lat": 1.270,
    "max_lat": 1.295,
    "min_lon": 103.840,
    "max_lon": 103.865,
}

ORCHARD_BOX = {
    "min_lat": 1.295,
    "max_lat": 1.315,
    "min_lon": 103.820,
    "max_lon": 103.840,
}

BUONA_VISTA_ONE_NORTH_BOX = {
    "min_lat": 1.290,
    "max_lat": 1.315,
    "min_lon": 103.780,
    "max_lon": 103.805,
}

PAYA_LEBAR_BOX = {
    "min_lat": 1.305,
    "max_lat": 1.330,
    "min_lon": 103.880,
    "max_lon": 103.905,
}

JURONG_EAST_BUSINESS_BOX = {
    "min_lat": 1.320,
    "max_lat": 1.345,
    "min_lon": 103.730,
    "max_lon": 103.755,
}

CHANGI_BUSINESS_PARK_BOX = {
    "min_lat": 1.325,
    "max_lat": 1.350,
    "min_lon": 103.955,
    "max_lon": 103.975,
}

HUB_BOXES = {
    "downtown_core": DOWNTOWN_CORE_BOX,
    "raffles_marina_cityhall": RAFFLES_MARINA_CITYHALL_BOX,
    "orchard": ORCHARD_BOX,
    "buona_vista_one_north": BUONA_VISTA_ONE_NORTH_BOX,
    "paya_lebar": PAYA_LEBAR_BOX,
    "jurong_east_business": JURONG_EAST_BUSINESS_BOX,
    "changi_business_park": CHANGI_BUSINESS_PARK_BOX,
}



def get_hub(lat, lon):
    """
    Returns the name of the hub whose bounding box contains this (lat, lon),
    or None if it doesn't belong to any.
    """
    for hub_name, box in HUB_BOXES.items():
        if (box["min_lat"] <= lat <= box["max_lat"] and
            box["min_lon"] <= lon <= box["max_lon"]):
            return hub_name
    return None


def generate():
    print(f"🔨 Reading {INPUT_GEOJSON}...")
    if not os.path.exists(INPUT_GEOJSON):
        print("❌ Error: Input file not found")
        return

    with open(INPUT_GEOJSON, 'r') as f:
        geo_data = json.load(f)
        
    predictions = []
    geometries = [] # List for the shapes
    processed_ids = set() # To ensure unique shapes
    
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Traffic Patterns
    traffic_pattern = { 7: 0.6, 8: 0.5, 9: 0.6, 12: 0.8, 13: 0.8, 17: 0.6, 18: 0.5, 19: 0.6, 23: 1.2, 0: 1.3 }

    features = geo_data.get('features', [])
    print(f"🚦 Processing {len(features)} segments...")

    for feature in features:
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        if geom.get('type') != 'LineString': continue
        coords = geom.get('coordinates', [])
        if not coords: continue

        start_lon, start_lat = coords[0]
        end_lon, end_lat = coords[-1]
        
        road_name = props.get('RD_CD_DESC') or props.get('RoadName') or 'Unknown'
        
        # --- CRITICAL FIX: Convert to STRING ---
        road_id = str(abs(hash(road_name + str(start_lon)))) 
        
        # --- 1. SAVE GEOMETRY ---
        if road_id not in processed_ids:
            geometries.append({
                "type": "Feature",
                # We don't need root ID if we use properties.id and promoteId
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "id": road_id, # <--- JS looks here for ID
                    "road_name": road_name
                }
            })
            processed_ids.add(road_id)

        # --- 2. SAVE DATA ---
        cat_raw = props.get('RD_CATG__1', '')
        if 'Category 1' in cat_raw or 'Expressway' in cat_raw: base_speed = 90
        elif 'Category 2' in cat_raw: base_speed = 60
        else: base_speed = 40 

        road_variance = random.uniform(0.8, 1.2)
        road_hub = get_hub(start_lat, start_lon)

        for h in range(HOURS_TO_PREDICT):
            future_time = start_time + timedelta(hours=h)
            hour_of_day = future_time.hour
            time_factor = traffic_pattern.get(hour_of_day, 1.0)
            
            if road_hub is not None:
                if hour_of_day in [8, 9, 18, 19]: time_factor *= 0.4
                elif hour_of_day in [7, 17]: time_factor *= 0.6
                elif hour_of_day in [12, 13]: time_factor *= 0.7

            noise = random.uniform(0.9, 1.1)
            final_speed = max(5.0, min(base_speed * time_factor * road_variance * noise, 120.0))
            
            ratio = final_speed / base_speed
            if ratio >= 0.8: color = "#00FF00"
            elif ratio >= 0.5: color = "#FFA500"
            else: color = "#FF0000"

            predictions.append({
                "id": road_id, # String ID
                "t": future_time.isoformat(), 
                "s": round(final_speed, 1),   
                "c": color                    
            })
    
    with open(OUTPUT_GEO_FILE, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": geometries}, f)
        
    with open(OUTPUT_DATA_FILE, 'w') as f:
        json.dump(predictions, f)
        
    print(f"✅ Done! Geometry: {len(geometries)} roads. Data: {len(predictions)} rows.")

if __name__ == "__main__":
    generate()