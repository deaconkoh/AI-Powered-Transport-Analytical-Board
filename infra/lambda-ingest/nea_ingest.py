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
    datagov_key = secrets.get("DATA_GOV_SG_API_KEY")
    headers = {"api-key": datagov_key} if datagov_key else {}

    # NEA API
    weather = http_get("https://api.data.gov.sg/v1/environment/air-temperature", headers)
    humidity = http_get("https://api.data.gov.sg/v1/environment/relative-humidity", headers)
    wind = http_get("https://api.data.gov.sg/v1/environment/wind-speed", headers)
    forecast = http_get("https://api.data.gov.sg/v1/environment/24-hour-weather-forecast", headers)

    latest = {
        "temperature": weather.get("items", [{}])[0].get("readings", [{}])[0].get("value"),
        "humidity": humidity.get("items", [{}])[0].get("readings", [{}])[0].get("value"),
        "wind_speed": wind.get("items", [{}])[0].get("readings", [{}])[0].get("value"),
        "forecast": forecast.get("items", [{}])[0].get("general", {}).get("forecast"),
        "resolved_area": "Singapore",
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    keypath = f"raw/nea/weather/{ts}.json"

    s3.put_object(
        Bucket=RAW_BUCKET,
        Key=keypath,
        Body=json.dumps(latest).encode("utf-8"),
        ContentType="application/json",
    )

    return {"ok": True, "file": keypath}
