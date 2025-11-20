import boto3
import time
import os

# Config
DATABASE = 'traffic_ai_db'
TABLE = 'hybrid'
REGION = 'us-east-1'

def get_latest_predictions(limit=2000): # Increased limit to cover more roads
    """
    Queries Athena to get the latest traffic predictions.
    """
    client = boto3.client('athena', region_name=REGION)
    
    # Get output location from Env Var
    output_location = os.environ.get('ATHENA_OUTPUT_LOCATION')
    
    if not output_location:
        print("❌ ATHENA_OUTPUT_LOCATION not set in environment variables")
        return []

    # --- THE FIX: SIMPLIFIED SQL ---
    # 1. Removed 'WHERE > now()' to prevent timezone issues returning 0 rows.
    # 2. Added casting to ensure numbers are returned as floats/strings correctly.
    query = f"""
    SELECT linkid, predictiontime, predictedspeed, startlatitude, startlongitude, endlatitude, endlongitude
    FROM "{DATABASE}"."{TABLE}"
    ORDER BY predictiontime DESC
    LIMIT {limit}
    """
    
    print(f"🚀 Running Query: {query}")

    try:
        # 1. Start Query
        response = client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': DATABASE},
            ResultConfiguration={'OutputLocation': output_location}
        )
        query_execution_id = response['QueryExecutionId']

        # 2. Wait for results
        while True:
            stats = client.get_query_execution(QueryExecutionId=query_execution_id)
            status = stats['QueryExecution']['Status']['State']
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(0.5) 
            
        if status == 'SUCCEEDED':
            # 3. Fetch Results
            results = client.get_query_results(QueryExecutionId=query_execution_id)
            
            # 4. Parse Results
            rows = []
            # Skip header row [0]
            for row in results['ResultSet']['Rows'][1:]:
                data = row['Data']
                try:
                    rows.append({
                        "link_id": data[0].get('VarCharValue'),
                        "time": data[1].get('VarCharValue'),
                        "speed": float(data[2].get('VarCharValue', 0)),
                        # Map new columns
                        "start_lat": float(data[3].get('VarCharValue', 0)),
                        "start_lon": float(data[4].get('VarCharValue', 0)),
                        "end_lat": float(data[5].get('VarCharValue', 0)),
                        "end_lon": float(data[6].get('VarCharValue', 0))
                    })
                except (ValueError, IndexError):
                    continue
                    
            print(f"✅ Athena returned {len(rows)} rows")
            return rows
        else:
            reason = stats['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            print(f"❌ Query failed: {reason}")
            return []

    except Exception as e:
        print(f"❌ Athena error: {e}")
        return []