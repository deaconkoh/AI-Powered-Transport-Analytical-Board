import os, json, time, urllib.request, boto3

RAW_BUCKET = os.environ["RAW_BUCKET"]
APP_SECRET_NAME = os.environ["APP_SECRET_NAME"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3")
sm = boto3.client("secretsmanager", region_name=REGION)

def get_secret():
    val = sm.get_secret_value(SecretId=APP_SECRET_NAME)
    return json.loads(val["SecretString"])

def http_get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def lambda_handler(event, context):
    secrets = get_secret()
    key = secrets.get("LTA_ACCOUNT_KEY")
    if not key:
        raise RuntimeError("Missing LTA_ACCOUNT_KEY in secret")

    headers = {"AccountKey": key}
    base = "https://datamall2.mytransport.sg/ltaodataservice"

    ts = time.strftime("%Y%m%d-%H%M%S")

    ##################################
    # Carpark availability
    ##################################
    cp_url = f"{base}/CarParkAvailabilityv2"
    carparks = http_get(cp_url, headers)

    carpark_payload = {
        "fetched_at": ts,
        "value": carparks.get("value", []),
    }

    carpark_keypath = f"raw/lta/carparks/{ts}.json"

    s3.put_object(
        Bucket=RAW_BUCKET,
        Key=carpark_keypath,
        Body=json.dumps(carpark_payload).encode("utf-8"),
        ContentType="application/json",
    )

    ##################################
    # Traffic Speed Bands (v4)
    ##################################
    speed_rows = []
    skip = 0
    PAGE_SIZE = 500
    SAFETY_LIMIT = 2000  # allows up to 1 million rows

    while True:
        sb_url = f"{base}/v4/TrafficSpeedBands?$skip={skip}"
        page = http_get(sb_url, headers)
        batch = page.get("value", [])

        if not batch:
            break  # no more pages

        speed_rows.extend(batch)

        # Continue pagination
        skip += PAGE_SIZE

        # Prevent infinite loops if API misbehaves
        if skip > PAGE_SIZE * SAFETY_LIMIT:
            print("WARNING: Safety limit reached, stopping pagination.")
            break

    speedband_payload = {
        "fetched_at": ts,
        "value": speed_rows,
    }

    speedband_keypath = f"raw/lta/speedbands/{ts}.json"

    s3.put_object(
        Bucket=RAW_BUCKET,
        Key=speedband_keypath,
        Body=json.dumps(speedband_payload).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "ok": True,
        "carpark_file": carpark_keypath,
        "carpark_count": len(carpark_payload["value"]),
        "speedband_file": speedband_keypath,
        "speedband_count": len(speedband_payload["value"]),
    }
