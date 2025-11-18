import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

class HybridTrafficModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, output_size=1):
        super(HybridTrafficModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=False)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.sum(lstm_out * attn_weights, dim=1)
        out = self.output(context)
        return out

class HybridProductionPredictor:
    def __init__(self, model_save_path=None):
        if model_save_path and os.path.isdir(model_save_path):
            self.model_path = os.path.join(model_save_path, 'final_hybrid_model.pth')
        elif model_save_path:
            self.model_path = model_save_path
        else:
            self.model_path = 'traffic_models_hybrid/final_hybrid_model.pth'
        
        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, 'final_hybrid_model.pth')
        
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained hybrid model"""
        print("Loading model from:", self.model_path)
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False
        
        if os.path.isdir(self.model_path):
            print(f"❌ Model path is a directory, not a file: {self.model_path}")
            return False
        
        try:
            loaded_data = torch.load(self.model_path, 
                                   weights_only=False,
                                   map_location=torch.device('cpu'))
            
            if isinstance(loaded_data, dict):
                print(f"📋 Loaded data keys: {list(loaded_data.keys())}")
                
                if 'lstm_state_dict' in loaded_data:
                    print("✅ Detected custom model format with lstm_state_dict")
                    self.model = HybridTrafficModel(input_size=11, hidden_size=64, output_size=1)
                    lstm_sd = loaded_data['lstm_state_dict']
                    self.model.load_state_dict(lstm_sd)
                    print("✅ Model loaded successfully from lstm_state_dict")
                else:
                    self.try_alternative_loading(loaded_data)
            else:
                self.model = loaded_data
                print("✅ Model loaded as nn.Module object")
            
            if self.model is not None:
                self.model.eval()
                print("✅ Model loaded and set to eval mode successfully")
                return True
            else:
                print("❌ Failed to load any model")
                return False
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def try_alternative_loading(self, loaded_data):
        """Try alternative methods to load the model"""
        if 'model' in loaded_data:
            self.model = loaded_data['model']
            print("✅ Loaded complete model from 'model' key")
            return
        
        if 'model_state_dict' in loaded_data:
            self.model = HybridTrafficModel(input_size=11, hidden_size=64, output_size=1)
            self.model.load_state_dict(loaded_data['model_state_dict'])
            print("✅ Loaded model from model_state_dict")
            return
        
        try:
            self.model = HybridTrafficModel(input_size=11, hidden_size=64, output_size=1)
            self.model.load_state_dict(loaded_data)
            print("✅ Loaded model from direct state dict")
            return
        except:
            pass
        
        print("❌ All alternative loading methods failed")
    
    def get_parquet_file(self):
        """Get the specific parquet file from test_data"""
        parquet_path = './test_data/gold_traffic_batch10_20251117_094330.parquet'
        
        if os.path.exists(parquet_path):
            print(f"✅ Found parquet file: {parquet_path}")
            return parquet_path
        else:
            print(f"❌ Parquet file not found: {parquet_path}")
            return None
    
    def preprocess_input(self, sample_size=1000):
        """Preprocess input data from the parquet file"""
        input_file = self.get_parquet_file()
        
        if input_file is None:
            print("❌ No input file available")
            return None, None
        
        print(f"Loading input data from: {input_file}")
        
        try:
            print(f"🔄 Sampling {sample_size} records to avoid memory issues...")
            
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows
            print(f"📊 Total records in file: {total_rows:,}")
            
            df = pd.read_parquet(input_file)
            
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"✅ Sampled {len(df)} records from {total_rows:,} total records")
            else:
                print(f"✅ Using all {len(df)} records")
            
            available_features = ['hour', 'day_of_week', 'is_weekend', 'AverageSpeed', 
                                'road_importance', 'optimal_speed', 'speed_efficiency',
                                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
            
            missing_features = [f for f in available_features if f not in df.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                print(f"Available features: {list(df.columns)}")
                
                df = self.create_missing_features(df, missing_features)
                
                missing_features = [f for f in available_features if f not in df.columns]
                if missing_features:
                    print(f"❌ Still missing features: {missing_features}")
                    return None, None
            
            features_df = df[available_features].copy()
            features_df = features_df.fillna(0)
            
            # Normalize features
            if 'hour' in features_df.columns:
                features_df['hour'] = features_df['hour'] / 23.0
            
            if 'day_of_week' in features_df.columns:
                features_df['day_of_week'] = features_df['day_of_week'] / 6.0
            
            if 'speed_efficiency' in features_df.columns:
                features_df['speed_efficiency'] = np.clip(features_df['speed_efficiency'], 0, 1.5)
            
            coord_columns = ['StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
            for col in coord_columns:
                if col in features_df.columns:
                    features_df[col] = np.clip(features_df[col], -180, 180)
                    features_df[col] = features_df[col] / 180.0
            
            print(f"✅ Selected {len(features_df)} records with {len(available_features)} features")
            print(f"📊 Features: {list(features_df.columns)}")
            
            return features_df, df
            
        except Exception as e:
            print(f"❌ Error loading input data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def create_missing_features(self, df, missing_features):
        """Create missing features from available data"""
        print("Attempting to create missing features...")
        
        for feature in missing_features:
            if feature == 'road_importance':
                print("  Creating default road_importance")
                df['road_importance'] = 1
            
            elif feature == 'optimal_speed':
                print("  Creating default optimal_speed")
                df['optimal_speed'] = 50
            
            elif feature == 'speed_efficiency' and 'AverageSpeed' in df.columns:
                print("  Creating speed_efficiency from AverageSpeed")
                df['speed_efficiency'] = df['AverageSpeed'] / 50
                df['speed_efficiency'] = np.clip(df['speed_efficiency'], 0, 1.5)
                df['speed_efficiency'] = df['speed_efficiency'].fillna(0)
            
            elif feature == 'is_weekend' and 'day_of_week' in df.columns:
                print("  Creating is_weekend from day_of_week")
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            else:
                print(f"  Cannot create {feature}, setting to 0")
                df[feature] = 0
        
        return df
    
    def prepare_model_input(self, features_df, sequence_length=10):
        """Convert features to model input format with sequences"""
        if features_df is None or len(features_df) == 0:
            return None, None
        
        max_sequences = 100
        features_array = features_df.values.astype(np.float32)
        
        sequences = []
        sequence_indices = []
        
        if len(features_array) >= sequence_length:
            for i in range(min(max_sequences, len(features_array) - sequence_length + 1)):
                sequence = features_array[i:i + sequence_length]
                sequences.append(sequence)
                sequence_indices.append(i + sequence_length - 1)
        else:
            padding = np.zeros((sequence_length - len(features_array), 11))
            padded_sequence = np.vstack([padding, features_array])
            sequences = [padded_sequence]
            sequence_indices.append(len(features_array) - 1)
        
        sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        print(f"✅ Prepared {len(sequences)} sequences of shape {sequences_tensor.shape}")
        return sequences_tensor, sequence_indices
    
    def run_prediction_pipeline(self):
        """Main prediction pipeline that returns data for frontend"""
        if self.model is None:
            print("❌ Model not loaded - cannot make predictions")
            return {"success": False, "error": "Model not loaded", "predictions": []}
        
        try:
            features_df, original_df = self.preprocess_input(sample_size=1000)
            if features_df is None or original_df is None:
                return {"success": False, "error": "No input data available", "predictions": []}
            
            input_tensor, sequence_indices = self.prepare_model_input(features_df)
            if input_tensor is None:
                return {"success": False, "error": "Could not prepare model input", "predictions": []}
            
            with torch.no_grad():
                predictions = self.model(input_tensor)
                predictions_np = predictions.numpy().flatten()
            
            print(f"✅ Generated {len(predictions_np)} predictions")
            print(f"📊 Prediction range: {predictions_np.min():.3f} to {predictions_np.max():.3f}")
            
            prediction_data = []
            for i, (pred, seq_idx) in enumerate(zip(predictions_np, sequence_indices)):
                original_row = original_df.iloc[seq_idx]
                
                prediction_obj = {
                    'link_id': f"pred_{i}",
                    'road_name': 'Predicted Road',
                    'predicted_speed': float(pred),
                    'current_speed': float(original_row.get('AverageSpeed', 50)),
                    'road_category': 'arterial',
                    'traffic_condition': self.get_traffic_condition(float(pred)),
                    'start_lon': float(original_row.get('StartLongitude', 103.8198)),
                    'start_lat': float(original_row.get('StartLatitude', 1.3521)),
                    'end_lon': float(original_row.get('EndLongitude', 103.8198)),
                    'end_lat': float(original_row.get('EndLatitude', 1.3521))
                }
                prediction_data.append(prediction_obj)
            
            result = {
                "success": True,
                "predictions": prediction_data,
                "count": len(prediction_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "predictions": []}
    
    def get_traffic_condition(self, speed):
        """Convert speed to traffic condition"""
        if speed > 60:
            return "Fluid"
        elif speed > 30:
            return "Moderate"
        else:
            return "Congested"
    
    def is_available(self):
        """Check if predictor is ready to use"""
        return self.model is not None