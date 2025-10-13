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
    data = {}

    # Fetch carpark availability
    url = f"{base}/CarParkAvailabilityv2"
    carparks = http_get(url, headers)
    data["value"] = carparks.get("value", [])

    ts = time.strftime("%Y%m%d-%H%M%S")
    keypath = f"raw/lta/carparks/{ts}.json"

    s3.put_object(
        Bucket=RAW_BUCKET,
        Key=keypath,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )

    return {"ok": True, "file": keypath}
