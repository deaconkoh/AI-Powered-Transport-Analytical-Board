import boto3
import time
import os

# Config
DATABASE = 'traffic_ai_db'
TABLE = 'hybrid'
REGION = 'us-east-1'

def get_latest_predictions(limit=2000):
    client = boto3.client('athena', region_name=REGION)
    output_location = os.environ.get('ATHENA_OUTPUT_LOCATION')
    
    if not output_location:
        print("❌ ATHENA_OUTPUT_LOCATION not set")
        return []
    
    # CORRECTED SQL: No Joins, using YOUR column names
    query = f"""
    SELECT 
        linkid, 
        predictiontime, 
        predictedspeed, 
        startlatitude, 
        startlongitude, 
        endlatitude, 
        endlongitude
    FROM "{DATABASE}"."{TABLE}"
    ORDER BY predictiontime ASC
    LIMIT {limit}
    """
    
    print(f"🚀 Running Query: {query}")

    try:
        response = client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': DATABASE},
            ResultConfiguration={'OutputLocation': output_location}
        )
        query_execution_id = response['QueryExecutionId']

        # Wait for results
        while True:
            stats = client.get_query_execution(QueryExecutionId=query_execution_id)
            status = stats['QueryExecution']['Status']['State']
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(0.5) 
            
        if status == 'SUCCEEDED':
            results = client.get_query_results(QueryExecutionId=query_execution_id)
            rows = []
            
            # Skip header [0]
            for row in results['ResultSet']['Rows'][1:]:
                d = row['Data']
                try:
                    
                    speed = float(d[2].get('VarCharValue', 0))
                    link_id = d[0].get('VarCharValue')
                    
                    # --- CALCULATE COLOR & STATUS ---
                    if speed >= 60:
                        color = "#00FF00" # Green
                        status = "Fluid"
                    elif speed >= 30:
                        color = "#FFA500" # Orange
                        status = "Moderate"
                    else:
                        color = "#FF0000" # Red
                        status = "Congested"

                    rows.append({
                        # Front-end expects 'link_id', DB gives 'linkid'
                        "link_id": link_id,
                        
                        # Front-end expects 'time', DB gives 'predictiontime'
                        "time": d[1].get('VarCharValue'),
                        
                        # Front-end expects 'speed', DB gives 'predictedspeed'
                        "speed": speed,
                        
                        # Front-end expects 'start_lat', DB gives 'startlatitude'
                        "start_lat": float(d[3].get('VarCharValue', 0)),
                        "start_lon": float(d[4].get('VarCharValue', 0)),
                        "end_lat": float(d[5].get('VarCharValue', 0)),
                        "end_lon": float(d[6].get('VarCharValue', 0)),
                        
                        # DB doesn't have road name, so we make one up or use ID
                        "road_name": f"Road {link_id}", 
                        
                        # Computed fields
                        "color": color,
                        "status": status,
                        "hub": "Unknown" 
                    })
                except (ValueError, IndexError) as e:
                    continue
                    
            print(f"✅ Athena returned {len(rows)} formatted rows")
            return rows
        else:
            print(f"❌ Query failed: {stats['QueryExecution']['Status'].get('StateChangeReason')}")
            return []

    except Exception as e:
        print(f"❌ Athena error: {e}")
        return []