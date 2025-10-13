"""
Traffic prediction using ensemble of models
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .model_loader import get_model_loader

class TrafficPredictor:
    def __init__(self):
        self.model_loader = get_model_loader()
        self.models = self.model_loader.load_models()
        self.device = self.model_loader.device
        print(f"TrafficPredictor initialized with {len(self.models)} models: {list(self.models.keys())}")
        
    def prepare_input_data(self, traffic_data: Dict, weather_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare input data for each model type with correct dimensions
        """
        try:
            print("Preparing input data for AI models...")
            
            # Extract basic features
            features = []
            
            # Add traffic features
            if 'traffic_speed_bands' in traffic_data:
                speeds = self._extract_speed_features(traffic_data['traffic_speed_bands'])
                features.extend(speeds)
                print(f"  - Added {len(speeds)} speed features")
                
            # Add weather features
            if 'air_temperature' in weather_data:
                temp = self._extract_temperature_features(weather_data['air_temperature'])
                features.extend(temp)
                print(f"  - Added {len(temp)} temperature features")
                
            if 'rainfall' in weather_data:
                rain = self._extract_rainfall_features(weather_data['rainfall'])
                features.extend(rain)
                print(f"  - Added {len(rain)} rainfall features")
            
            # Convert to numpy array for easier manipulation
            features_array = np.array(features, dtype=np.float32)
            print(f"Total features extracted: {len(features)}")
            
            # Prepare inputs for each model type with correct dimensions
            input_data = {}
            
            # Custom STGNN expects: (batch_size, seq_len, in_feats) = (1, 12, 108)
            if 'custom' in self.models:
                try:
                    # Pad/truncate to 108 features
                    custom_features = self._pad_features(features_array, 108)
                    # Create sequence of 12 time steps (repeat same features for demo)
                    custom_input = np.tile(custom_features, (12, 1))  # (12, 108)
                    custom_input = custom_input.reshape(1, 12, 108)  # (1, 12, 108)
                    input_data['custom'] = torch.tensor(custom_input, dtype=torch.float32).to(self.device)
                    print(f"Custom model input shape: {input_data['custom'].shape}")
                except Exception as e:
                    print(f"Error preparing custom model input: {e}")
                    input_data['custom'] = None
            
            # STGCN expects: (batch_size, seq_len, num_nodes, in_channels) = (1, 12, 100, 9)
            if 'stgcn' in self.models:
                try:
                    # Pad/truncate to 9 features
                    stgcn_features = self._pad_features(features_array, 9)
                    # Create input: (1, 12, 100, 9)
                    stgcn_input = np.zeros((1, 12, 100, 9), dtype=np.float32)
                    # Use the features for all nodes (simplified)
                    for node in range(100):
                        stgcn_input[0, :, node, :] = stgcn_features.reshape(1, 9)
                    input_data['stgcn'] = torch.tensor(stgcn_input, dtype=torch.float32).to(self.device)
                    print(f"STGCN model input shape: {input_data['stgcn'].shape}")
                except Exception as e:
                    print(f"Error preparing STGCN model input: {e}")
                    input_data['stgcn'] = None
            
            # Graph WaveNet expects: (batch_size, seq_len, num_nodes, in_feats) = (1, 12, 100, 9)
            if 'graphwavenet' in self.models:
                try:
                    # Pad/truncate to 9 features
                    gwn_features = self._pad_features(features_array, 9)
                    # Create input: (1, 12, 100, 9)
                    gwn_input = np.zeros((1, 12, 100, 9), dtype=np.float32)
                    # Use the features for all nodes (simplified)
                    for node in range(100):
                        gwn_input[0, :, node, :] = gwn_features.reshape(1, 9)
                    input_data['graphwavenet'] = torch.tensor(gwn_input, dtype=torch.float32).to(self.device)
                    print(f"Graph WaveNet model input shape: {input_data['graphwavenet'].shape}")
                except Exception as e:
                    print(f"Error preparing Graph WaveNet model input: {e}")
                    input_data['graphwavenet'] = None
            
            return input_data
            
        except Exception as e:
            print(f"Error preparing input data: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _pad_features(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Pad or truncate features to target size"""
        if len(features) >= target_size:
            return features[:target_size]
        else:
            # Pad with zeros
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def predict_traffic_conditions(self, traffic_data: Dict, weather_data: Dict) -> Dict:
        """
        Make ensemble prediction using all three models
        """
        print(f"Starting prediction with {len(self.models)} available models")
        
        if not self.models:
            print("No models available for prediction")
            return {"error": "Models not loaded"}
            
        try:
            input_data = self.prepare_input_data(traffic_data, weather_data)
            if not input_data:
                print("Could not prepare input data")
                return {"error": "Could not prepare input data"}
                
            predictions = {}
            
            with torch.no_grad():
                # Get predictions from each model
                print("Running model predictions...")
                
                # Custom model prediction
                if 'custom' in self.models and input_data.get('custom') is not None:
                    try:
                        print("  - Running Custom STGNN...")
                        custom_pred = self.models['custom'](input_data['custom'])
                        # Handle different output formats
                        if isinstance(custom_pred, tuple):
                            custom_pred = custom_pred[0]  # Take first output if tuple
                        predictions['custom'] = custom_pred.cpu().numpy().tolist()
                        print(f"Custom prediction shape: {custom_pred.shape}")
                    except Exception as e:
                        print(f"Custom model failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  - Custom model not available or no input data")
                
                # STGCN prediction
                if 'stgcn' in self.models and input_data.get('stgcn') is not None:
                    try:
                        print("  - Running STGCN...")
                        # Create simple graph data
                        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
                        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
                        
                        stgcn_pred = self.models['stgcn'](input_data['stgcn'], edge_index, edge_weight)
                        # Handle different output formats
                        if isinstance(stgcn_pred, tuple):
                            stgcn_pred = stgcn_pred[0]
                        predictions['stgcn'] = stgcn_pred.cpu().numpy().tolist()
                        print(f"STGCN prediction shape: {stgcn_pred.shape}")
                    except Exception as e:
                        print(f"STGCN model failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  - STGCN model not available or no input data")
                
                # Graph WaveNet prediction
                if 'graphwavenet' in self.models and input_data.get('graphwavenet') is not None:
                    try:
                        print("  - Running Graph WaveNet...")
                        # Create simple graph data
                        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
                        edge_attr = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
                        
                        graphwavenet_pred = self.models['graphwavenet'](input_data['graphwavenet'], edge_index, edge_attr)
                        # Handle different output formats
                        if isinstance(graphwavenet_pred, tuple):
                            graphwavenet_pred = graphwavenet_pred[0]
                        predictions['graphwavenet'] = graphwavenet_pred.cpu().numpy().tolist()
                        print(f"Graph WaveNet prediction shape: {graphwavenet_pred.shape}")
                    except Exception as e:
                        print(f"Graph WaveNet model failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  - Graph WaveNet model not available or no input data")
            
            print(f"📊 Models that produced predictions: {list(predictions.keys())}")
            
            # Ensemble prediction
            ensemble_pred = None
            if predictions:
                print("Computing ensemble prediction...")
                # Use the weights: 0.5 GraphWaveNet, 0.3 STGCN, 0.2 Custom
                weights = {'graphwavenet': 0.5, 'stgcn': 0.3, 'custom': 0.2}
                
                for model_name, pred_array in predictions.items():
                    weight = weights.get(model_name, 0)
                    pred_np = np.array(pred_array)
                    
                    # Flatten the prediction for ensemble
                    flat_pred = pred_np.flatten()
                    
                    if ensemble_pred is None:
                        ensemble_pred = weight * flat_pred
                    else:
                        # Ensure shapes match by taking the minimum length
                        min_len = min(len(ensemble_pred), len(flat_pred))
                        ensemble_pred[:min_len] += weight * flat_pred[:min_len]
                
                if ensemble_pred is not None:
                    predictions['ensemble'] = ensemble_pred.tolist()
                    print(f"Ensemble prediction computed with {len(ensemble_pred)} values")
                else:
                    print("Could not compute ensemble prediction")
            else:
                print("No models produced predictions")
            
            # Create fallback prediction if no models worked
            if not predictions:
                print("Using fallback prediction based on traffic data")
                fallback_pred = self._create_fallback_prediction(traffic_data)
                predictions['fallback'] = fallback_pred
                ensemble_pred = np.array(fallback_pred)
            
            result = {
                "predictions": predictions,
                "confidence": self._calculate_confidence(ensemble_pred) if ensemble_pred is not None else 0.5,
                "timestamp": np.datetime64('now').astype(str)
            }
            
            print(f"Final prediction result has {len(predictions)} model outputs")
            return result
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _create_fallback_prediction(self, traffic_data: Dict) -> List[float]:
        """Create a simple fallback prediction based on available traffic data"""
        print("Creating fallback prediction...")
        
        # Simple logic based on traffic incidents and speed bands
        base_prediction = [0.7, 0.6, 0.8, 0.5, 0.9]  # Default moderate traffic
        
        try:
            # Adjust based on incidents
            incidents = traffic_data.get('incidents', {}).get('value', [])
            if incidents:
                incident_count = len(incidents)
                if incident_count > 10:
                    base_prediction = [0.3, 0.4, 0.2, 0.5, 0.3]  # Heavy traffic
                elif incident_count > 5:
                    base_prediction = [0.5, 0.6, 0.4, 0.7, 0.5]  # Moderate-heavy traffic
            
            # Adjust based on speed bands
            speed_bands = traffic_data.get('traffic_speed_bands', {}).get('value', [])
            if speed_bands:
                slow_segments = sum(1 for band in speed_bands if band.get('SpeedBand', 0) < 50)
                total_segments = len(speed_bands)
                if total_segments > 0:
                    slow_ratio = slow_segments / total_segments
                    if slow_ratio > 0.3:
                        base_prediction = [p * 0.7 for p in base_prediction]  # Reduce prediction values
        except Exception as e:
            print(f"Error in fallback prediction: {e}")
        
        return base_prediction
    
    def _extract_speed_features(self, speed_data: Dict) -> List[float]:
        """Extract speed-related features from traffic data"""
        features = []
        if 'value' in speed_data:
            for item in speed_data['value'][:20]:  # First 20 segments
                if 'SpeedBand' in item:
                    try:
                        features.append(float(item['SpeedBand']) / 100.0)  # Normalize
                    except (ValueError, TypeError):
                        continue
        return features if features else [0.5] * 10  # Default moderate speed
    
    def _extract_temperature_features(self, temp_data: Dict) -> List[float]:
        """Extract temperature features"""
        features = []
        if 'items' in temp_data and temp_data['items']:
            readings = temp_data['items'][0].get('readings', [])
            for reading in readings[:5]:  # First 5 stations
                try:
                    temp = float(reading.get('value', 0))
                    features.append(temp / 50.0)  # Normalize (assuming max 50°C)
                except (ValueError, TypeError):
                    continue
        return features if features else [0.6] * 5  # Default ~30°C
    
    def _extract_rainfall_features(self, rain_data: Dict) -> List[float]:
        """Extract rainfall features"""
        features = []
        if 'items' in rain_data and rain_data['items']:
            readings = rain_data['items'][0].get('readings', [])
            for reading in readings[:5]:  # First 5 stations
                try:
                    rain = float(reading.get('value', 0))
                    features.append(min(rain / 10.0, 1.0))  # Normalize (assuming max 10mm)
                except (ValueError, TypeError):
                    continue
        return features if features else [0.1] * 5  # Default light rain
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        if prediction is not None and len(prediction) > 0:
            # Simple confidence based on prediction variance
            variance = np.var(prediction)
            confidence = max(0.1, 1.0 - variance)
            return float(confidence)
        return 0.5  # Default confidence

# Global instance
_predictor = None

def get_traffic_predictor():
    global _predictor
    if _predictor is None:
        _predictor = TrafficPredictor()
    return _predictor