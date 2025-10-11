"""
Route prediction integration with existing routing system
"""
from typing import Dict, Any, Optional, Tuple, List
from ..models.traffic_predictor import get_traffic_predictor
from ..data.lta_client import client as lta_client
from ..data.weather_client import get_weather_data

class RoutePredictor:
    def __init__(self):
        self.traffic_predictor = get_traffic_predictor()
        self.lta_client = lta_client()
        self.prediction_count = 0  # Track how many predictions we've made
        
    def predict_route_conditions(self, origin: Tuple[float, float], 
                               destination: Tuple[float, float]) -> Dict[str, Any]:
        """
        Predict traffic conditions for a route
        """
        self.prediction_count += 1
        print(f"Making AI prediction #{self.prediction_count} from {origin} to {destination}")
        
        try:
            # Get real-time data
            print("Fetching real-time traffic and weather data...")
            traffic_data = self._get_traffic_data()
            weather_data = get_weather_data()
            
            # Get traffic prediction
            print("Running AI model predictions...")
            prediction = self.traffic_predictor.predict_traffic_conditions(
                traffic_data, weather_data
            )
            
            # Check if models were actually used
            if "error" in prediction:
                print(f"Prediction failed: {prediction['error']}")
            else:
                print(f"AI prediction successful! Confidence: {prediction.get('confidence', 'N/A')}")
                if "predictions" in prediction:
                    model_used = list(prediction["predictions"].keys())
                    print(f"📈 Models used in ensemble: {model_used}")
            
            # Enhance route prediction with AI insights
            enhanced_prediction = self._enhance_route_prediction(prediction, origin, destination)
            
            return enhanced_prediction
            
        except Exception as e:
            print(f"Route prediction failed: {e}")
            return {"error": f"Route prediction failed: {str(e)}"}
    
    def _get_traffic_data(self) -> Dict[str, Any]:
        """Get comprehensive traffic data from LTA APIs"""
        traffic_data = {}
        
        try:
            print("Fetching traffic incidents...")
            traffic_data['incidents'] = self.lta_client.traffic_incidents()
            
            print("Fetching traffic speed bands...")
            traffic_data['traffic_speed_bands'] = self.lta_client.traffic_speed_bands()
            
            print("Fetching travel times...")
            traffic_data['travel_times'] = self.lta_client.est_travel_times()
            
            print(f"Fetched {len(traffic_data.get('incidents', {}).get('value', []))} incidents, "
                  f"{len(traffic_data.get('traffic_speed_bands', {}).get('value', []))} speed segments")
            
        except Exception as e:
            print(f"Could not fetch some traffic data: {e}")
            
        return traffic_data
    
    def _enhance_route_prediction(self, prediction: Dict, 
                                origin: Tuple[float, float],
                                destination: Tuple[float, float]) -> Dict[str, Any]:
        """Enhance basic route prediction with AI insights"""
        
        enhanced = {
            "route": {
                "origin": origin,
                "destination": destination,
                "prediction_timestamp": prediction.get("timestamp"),
            },
            "traffic_conditions": {
                "overall_confidence": prediction.get("confidence", 0.5),
                "predicted_congestion_level": self._calculate_congestion_level(prediction),
                "recommended_departure_time": self._calculate_optimal_departure(prediction),
            },
            "ai_insights": {
                "model_contributions": {
                    "graphwavenet": 0.5,
                    "stgcn": 0.3, 
                    "custom": 0.2
                },
                "risk_factors": self._identify_risk_factors(prediction),
                "prediction_id": self.prediction_count  # Track which prediction this is
            },
            "models_used": list(prediction.get("predictions", {}).keys()) if "predictions" in prediction else []
        }
        
        # Add raw predictions if available (for debugging)
        if "predictions" in prediction:
            enhanced["raw_predictions_summary"] = {
                model: f"{len(preds)} values" 
                for model, preds in prediction["predictions"].items()
            }
            
        return enhanced
    
    def _calculate_congestion_level(self, prediction: Dict) -> str:
        """Calculate congestion level from prediction"""
        ensemble_pred = prediction.get("predictions", {}).get("ensemble", [])
        if ensemble_pred:
            avg_speed = sum(ensemble_pred) / len(ensemble_pred)
            if avg_speed > 0.8:
                return "LIGHT"
            elif avg_speed > 0.5:
                return "MODERATE"
            else:
                return "HEAVY"
        return "MODERATE"
    
    def _calculate_optimal_departure(self, prediction: Dict) -> str:
        """Calculate optimal departure time"""
        import datetime
        now = datetime.datetime.now()
        
        congestion = self._calculate_congestion_level(prediction)
        if congestion == "HEAVY":
            optimal_time = now + datetime.timedelta(minutes=30)
            return optimal_time.strftime("%H:%M")
        else:
            return "NOW"
    
    def _identify_risk_factors(self, prediction: Dict) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        congestion = self._calculate_congestion_level(prediction)
        if congestion == "HEAVY":
            risk_factors.append("Heavy traffic congestion predicted")
        
        confidence = prediction.get("confidence", 0)
        if confidence < 0.6:
            risk_factors.append("Low prediction confidence")
            
        ensemble_pred = prediction.get("predictions", {}).get("ensemble", [])
        if ensemble_pred:
            max_pred = max(ensemble_pred)
            min_pred = min(ensemble_pred)
            if max_pred - min_pred > 0.5:
                risk_factors.append("Unstable traffic conditions")
                
        return risk_factors if risk_factors else ["No significant risks identified"]

# Global instance
_route_predictor = None

def get_route_predictor():
    global _route_predictor
    if _route_predictor is None:
        _route_predictor = RoutePredictor()
    return _route_predictor