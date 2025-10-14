import os, json, boto3
from typing import Any, Dict
from traffic_predictor import TrafficPredictor

session = boto3.session.Session()
region = os.environ.get("AWS_DEFAULT_REGION") or session.region_name or "us-east-1"
sec = boto3.client("secretsmanager", region_name=region)

secret_name = os.environ.get("APP_SECRET_NAME")
if not secret_name:
    raise RuntimeError("APP_SECRET_NAME not set in environment")

blob = sec.get_secret_value(SecretId=secret_name)["SecretString"]
s = json.loads(blob)
os.environ["LTA_ACCOUNT_KEY"] = s.get("LTA_ACCOUNT_KEY","")
os.environ["DATA_GOV_SG_API_KEY"] = s.get("DATA_GOV_SG_API_KEY","")


# SageMaker calls this once when the container starts or the model is loaded
def model_fn(model_dir: str):
    import os, shutil
    print(">>> model_dir:", model_dir)
    wdir = os.path.join(model_dir, "weights")
    # Ensure /opt/models exists and contains the weights the predictor expects
    os.makedirs("/opt/models", exist_ok=True)
    if os.path.isdir(wdir):
        # Python 3.8+: dirs_exist_ok
        for name in os.listdir(wdir):
            src = os.path.join(wdir, name)
            dst = os.path.join("/opt/models", name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(">>> Copied weights to /opt/models")
    else:
        print(">>> WARNING: weights/ not found inside model_dir")

    # Set both env vars, in case your code reads either
    os.environ["MODEL_DIR"] = "/opt/models"
    os.environ["MODELS_DIR"] = "/opt/models"

    predictor = TrafficPredictor()
    print(">>> TrafficPredictor constructed OK")
    return predictor

# Parse the incoming HTTP request body
def input_fn(request_body: bytes, content_type: str):
    return json.loads(request_body)

# Do the prediction using the API
def predict_fn(data: Dict[str, Any], predictor: TrafficPredictor):
    # If the request comes from /predict_route: use origin/destination
    if "origin" in data and "destination" in data:
        origin = tuple(data["origin"])
        destination = tuple(data["destination"])
        # whichever method your predictor exposes:
        if hasattr(predictor, "predict_route_conditions"):
            return predictor.predict_route_conditions(origin, destination)
        # fallback – or map into your universal API if you have one
        raise ValueError("predict_route_conditions() not found on predictor")

    # Else: support the old shape (traffic_data/weather_data)
    traffic_data = data.get("traffic_data", {})
    weather_data = data.get("weather_data", {})
    if hasattr(predictor, "predict_traffic_conditions"):
        return predictor.predict_traffic_conditions(traffic_data, weather_data)

    raise ValueError("No supported prediction method found")

# Serialize the response
def output_fn(prediction, accept: str):
    return json.dumps(prediction), "application/json"
