import os, json
from typing import Any, Dict
from traffic_predictor import TrafficPredictor

# SageMaker calls this once when the container starts or the model is loaded
def model_fn(model_dir: str):
    """
    model_dir will contain model artifact (e.g., weights in model_dir/weights).
    We expose MODEL_DIR so model_loader can find the .pth files.
    """
    os.environ["MODEL_DIR"] = os.path.join(model_dir, "weights")
    predictor = TrafficPredictor()  # your class loads models internally
    return predictor

# Parse the incoming HTTP request body
def input_fn(request_body: bytes, content_type: str):
    return json.loads(request_body)

# Do the prediction using the API
def predict_fn(data: Dict[str, Any], predictor: TrafficPredictor):
    traffic_data = data.get("traffic_data", {})
    weather_data = data.get("weather_data", {})
    return predictor.predict_traffic_conditions(traffic_data, weather_data)

# Serialize the response
def output_fn(prediction, accept: str):
    return json.dumps(prediction), "application/json"
