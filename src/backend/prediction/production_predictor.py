import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

class TrafficPredictor(torch.nn.Module):
    """Neural Network for Traffic Prediction - SAME ARCHITECTURE AS TRAINING"""
    def __init__(self, input_size, hidden_layers=[512, 256, 128, 64], dropout_rate=0.4):
        super(TrafficPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(prev_size, 1)
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class ProductionPredictor:
    def __init__(self, model_path, recent_data_dir="gold_recent_24h"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.recent_data_dir = recent_data_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model for production use"""
        print("Loading production model...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model data
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Failed to load with weights_only=True: {e}")
            print("Trying with weights_only=False...")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Load preprocessing objects
        preprocess_path = self.model_path.replace('.pth', '_preprocess.joblib')
        if not os.path.exists(preprocess_path):
            raise FileNotFoundError(f"Preprocessing file not found: {preprocess_path}")
        
        preprocess_data = joblib.load(preprocess_path)
        self.scaler = preprocess_data['scaler']
        self.label_encoders = preprocess_data['label_encoders']
        
        # Recreate model
        self.model = TrafficPredictor(
            input_size=len(checkpoint['feature_names']),
            hidden_layers=checkpoint['config']['hidden_layers'],
            dropout_rate=checkpoint['config']['dropout_rate']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_names = checkpoint['feature_names']
        
        print(f"Production model loaded: {len(self.feature_names)} features")
        print(f"Device: {self.device}")
    
    def load_recent_data(self):
        """Load the most recent 24-hour data"""
        recent_files = glob.glob(os.path.join(self.recent_data_dir, "recent_gold_*.csv"))
        if not recent_files:
            raise FileNotFoundError("No recent data files found")
        
        # Get the most recent file
        latest_file = max(recent_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Ensure datetime
        if 'Retrieval_Time' in df.columns:
            df['Retrieval_Time'] = pd.to_datetime(df['Retrieval_Time'])
        
        print(f"Loaded recent data: {len(df):,} records from {os.path.basename(latest_file)}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for prediction (SAME as training preprocessing)"""
        print("Preparing features for prediction...")
        
        # Select available features
        available_features = [col for col in self.feature_names if col in df.columns]
        features_df = df[available_features].copy()
        
        # Handle missing features
        missing_features = set(self.feature_names) - set(available_features)
        if missing_features:
            print(f"Missing features: {missing_features}")
            for feature in missing_features:
                features_df[feature] = 0  # Fill with zeros
        
        # Handle categorical variables
        for col in self.label_encoders:
            if col in features_df.columns:
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                current_categories = set(features_df[col].unique())
                unseen_categories = current_categories - known_categories
                
                if unseen_categories:
                    most_frequent = self.label_encoders[col].classes_[0]
                    features_df[col] = features_df[col].apply(
                        lambda x: x if x in known_categories else most_frequent
                    )
                
                features_df[col] = self.label_encoders[col].transform(features_df[col])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df.values)
        
        print(f"Features prepared: {len(available_features)} features")
        return features_scaled
    
    def predict(self, features):
        """Make predictions"""
        print("Making predictions...")
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(features_tensor).squeeze().cpu().numpy()
        
        print(f"Predictions completed: {len(predictions):,} predictions")
        return predictions
    
    def create_heatmap_data(self, original_df, predictions):
        """Create heatmap data with coordinates and predicted speeds"""
        print("Creating heatmap data...")
        
        heatmap_data = []
        for idx, row in original_df.iterrows():
            # Get road segment coordinates
            start_lon = row.get('StartLongitude', 0)
            start_lat = row.get('StartLatitude', 0)
            end_lon = row.get('EndLongitude', 0)
            end_lat = row.get('EndLatitude', 0)
            
            # Skip if no coordinates
            if start_lon == 0 and start_lat == 0 and end_lon == 0 and end_lat == 0:
                continue
            
            heatmap_data.append({
                'link_id': row.get('LinkID', ''),
                'road_name': row.get('RoadName', ''),
                'start_lon': float(start_lon),
                'start_lat': float(start_lat),
                'end_lon': float(end_lon),
                'end_lat': float(end_lat),
                'predicted_speed': float(predictions[idx]),
                'current_speed': float(row.get('AverageSpeed', 0)),
                'road_category': row.get('RoadCategory_Description', ''),
                'traffic_condition': row.get('traffic_condition', ''),
                'timestamp': row.get('Retrieval_Time', '').strftime('%Y-%m-%d %H:%M:%S') if hasattr(row.get('Retrieval_Time', ''), 'strftime') else str(row.get('Retrieval_Time', ''))
            })
        
        print(f"Heatmap data created: {len(heatmap_data)} road segments")
        return heatmap_data
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline"""
        print("Starting Production Prediction Pipeline")
        print("=" * 50)
        
        try:
            # 1. Load recent data
            recent_data = self.load_recent_data()
            
            if recent_data.empty:
                print("No recent data available for prediction")
                return None
            
            # 2. Prepare features
            features = self.prepare_features(recent_data)
            
            # 3. Make predictions
            predictions = self.predict(features)
            
            # 4. Create heatmap data
            heatmap_data = self.create_heatmap_data(recent_data, predictions)
            
            # 5. Save predictions
            output_file = "current_traffic_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
            
            # 6. Print summary
            avg_predicted_speed = np.mean(predictions)
            avg_current_speed = recent_data['AverageSpeed'].mean()
            
            print(f"PREDICTION SUMMARY:")
            print(f"   Road segments predicted: {len(heatmap_data):,}")
            print(f"   Average predicted speed: {avg_predicted_speed:.1f} km/h")
            print(f"   Average current speed: {avg_current_speed:.1f} km/h")
            print(f"   Predictions saved to: {output_file}")
            
            return heatmap_data
            
        except Exception as e:
            print(f"Prediction pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function for production predictions"""
    try:
        # Initialize predictor
        predictor = ProductionPredictor(
            model_path="traffic_models/traffic_predictor_final.pth",
            recent_data_dir="gold_recent_24h"
        )
        
        # Run prediction pipeline
        results = predictor.run_prediction_pipeline()
        
        if results:
            print("Production prediction completed successfully!")
            print("Data is ready for heatmap visualization")
        else:
            print("Production prediction failed")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

if __name__ == "__main__":
    main()