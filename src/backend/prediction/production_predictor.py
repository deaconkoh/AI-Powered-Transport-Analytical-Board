import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import json
import joblib
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        print("Calculating performance metrics...")
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100  # Avoid division by zero
        rmse_mae_ratio = rmse / mae if mae != 0 else float('inf')
        
        # Error distribution metrics
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        
        # Percentage within different error ranges
        within_5_kmh = np.mean(np.abs(errors) <= 5) * 100
        within_10_kmh = np.mean(np.abs(errors) <= 10) * 100
        within_20_kmh = np.mean(np.abs(errors) <= 20) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2,
            'MAPE': mape,
            'RMSE_MAE_Ratio': rmse_mae_ratio,
            'Mean_Error': mean_error,
            'Std_Error': std_error,
            'Max_Absolute_Error': max_error,
            'Within_5_kmh_Percent': within_5_kmh,
            'Within_10_kmh_Percent': within_10_kmh,
            'Within_20_kmh_Percent': within_20_kmh,
            'Sample_Count': len(y_true)
        }
        
        return metrics
    
    def print_metrics_report(self, metrics):
        """Print a comprehensive metrics report"""
        print("\n" + "="*70)
        print("PREDICTION PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\nCORE METRICS:")
        print(f"   * R² Score:           {metrics['R2_Score']:.4f}")
        print(f"   * MAE (Mean Abs Error): {metrics['MAE']:.2f} km/h")
        print(f"   * RMSE (Root Mean Sq): {metrics['RMSE']:.2f} km/h")
        print(f"   * MAPE:               {metrics['MAPE']:.2f}%")
        
        print(f"\nERROR ANALYSIS:")
        print(f"   * RMSE/MAE Ratio:     {metrics['RMSE_MAE_Ratio']:.2f}")
        if metrics['RMSE_MAE_Ratio'] > 1.5:
            print("     Note: High ratio indicates significant large errors")
        else:
            print("     Note: Ratio suggests relatively uniform error distribution")
        
        print(f"   * Mean Error:         {metrics['Mean_Error']:.2f} km/h")
        print(f"   * Error Std Dev:      {metrics['Std_Error']:.2f} km/h")
        print(f"   * Max Absolute Error: {metrics['Max_Absolute_Error']:.2f} km/h")
        
        print(f"\nACCURACY THRESHOLDS:")
        print(f"   * Within ±5 km/h:     {metrics['Within_5_kmh_Percent']:.1f}% of predictions")
        print(f"   * Within ±10 km/h:    {metrics['Within_10_kmh_Percent']:.1f}% of predictions")
        print(f"   * Within ±20 km/h:    {metrics['Within_20_kmh_Percent']:.1f}% of predictions")
        
        print(f"\nSUMMARY:")
        print(f"   * Total Predictions:  {metrics['Sample_Count']:,}")
        print(f"   * Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance interpretation
        print(f"\nPERFORMANCE INTERPRETATION:")
        if metrics['R2_Score'] >= 0.8:
            print("   [EXCELLENT] Excellent explanatory power")
        elif metrics['R2_Score'] >= 0.6:
            print("   [GOOD] Good model performance")
        elif metrics['R2_Score'] >= 0.4:
            print("   [MODERATE] Moderate performance")
        else:
            print("   [NEEDS IMPROVEMENT] Model may need improvement")
            
        if metrics['MAPE'] <= 10:
            print("   [EXCELLENT] Excellent prediction accuracy")
        elif metrics['MAPE'] <= 20:
            print("   [GOOD] Good prediction accuracy")
        elif metrics['MAPE'] <= 30:
            print("   [ACCEPTABLE] Acceptable accuracy for traffic prediction")
        else:
            print("   [NEEDS IMPROVEMENT] Accuracy may need improvement")
        
        print("="*70)
    
    def save_metrics_report(self, metrics, output_file="prediction_metrics_report.json"):
        """Save metrics to JSON file for tracking over time"""
        # Add timestamp
        metrics_with_timestamp = metrics.copy()
        metrics_with_timestamp['prediction_timestamp'] = datetime.now().isoformat()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=2)
        
        print(f"\nMetrics report saved to: {output_file}")
    
    def create_error_visualization(self, y_true, y_pred, save_path="error_analysis.png"):
        """Create visualization of prediction errors (optional)"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Actual vs Predicted
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Speed (km/h)')
            plt.ylabel('Predicted Speed (km/h)')
            plt.title('Actual vs Predicted Speeds')
            
            # Plot 2: Error distribution
            plt.subplot(2, 2, 2)
            errors = y_pred - y_true
            plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Error (km/h)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            
            # Plot 3: Residual plot
            plt.subplot(2, 2, 3)
            plt.scatter(y_pred, errors, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Speed (km/h)')
            plt.ylabel('Residual Error (km/h)')
            plt.title('Residual Plot')
            
            # Plot 4: Error by speed range
            plt.subplot(2, 2, 4)
            speed_bins = np.linspace(y_true.min(), y_true.max(), 10)
            bin_errors = []
            for i in range(len(speed_bins)-1):
                mask = (y_true >= speed_bins[i]) & (y_true < speed_bins[i+1])
                if np.sum(mask) > 0:
                    bin_errors.append(np.mean(np.abs(errors[mask])))
                else:
                    bin_errors.append(0)
            
            plt.bar(range(len(bin_errors)), bin_errors, alpha=0.7)
            plt.xlabel('Speed Range Bin')
            plt.ylabel('Mean Absolute Error (km/h)')
            plt.title('Error by Speed Range')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Error visualization saved to: {save_path}")
            
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
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
            
            # 4. Calculate metrics (using current speed as ground truth)
            y_true = recent_data['AverageSpeed'].values
            metrics = self.calculate_metrics(y_true, predictions)
            
            # 5. Print comprehensive report
            self.print_metrics_report(metrics)
            
            # 6. Save metrics report
            self.save_metrics_report(metrics)
            
            # 7. Create error visualization (optional)
            self.create_error_visualization(y_true, predictions)
            
            # 8. Create heatmap data
            heatmap_data = self.create_heatmap_data(recent_data, predictions)
            
            # 9. Save predictions
            output_file = "current_traffic_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
            
            # 10. Final summary
            print(f"\nPREDICTION SUMMARY:")
            print(f"   Road segments predicted: {len(heatmap_data):,}")
            print(f"   Average predicted speed: {np.mean(predictions):.1f} km/h")
            print(f"   Average current speed: {np.mean(y_true):.1f} km/h")
            print(f"   Model Performance: R² = {metrics['R2_Score']:.3f}, MAE = {metrics['MAE']:.2f} km/h")
            print(f"   Predictions saved to: {output_file}")
            
            return {
                'heatmap_data': heatmap_data,
                'metrics': metrics,
                'predictions': predictions,
                'actual_speeds': y_true
            }
            
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
            print("\nProduction prediction completed successfully!")
            print("Comprehensive metrics report generated")
            print("Data is ready for heatmap visualization")
        else:
            print("Production prediction failed")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

if __name__ == "__main__":
    main()