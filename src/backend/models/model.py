import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
import glob
import pickle
import joblib  # Alternative to pickle for sklearn objects
warnings.filterwarnings('ignore')

# Try to import from your existing tiers
try:
    from gold import GoldFeatureEngineer
except ImportError:
    print("⚠️ GoldFeatureEngineer not available, using mock data for testing")

class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TrafficPredictor(nn.Module):
    """Neural Network for Traffic Prediction"""
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        super(TrafficPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class ModelTrainer:
    def __init__(self, model_dir="traffic_models"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model configuration
        self.config = {
            'batch_size': 256,
            'learning_rate': 0.001,
            'epochs': 200,
            'patience': 25,
            'hidden_layers': [512, 256, 128, 64],
            'dropout_rate': 0.4,
            'validation_split': 0.15,
            'test_split': 0.15
        }
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.optimizer = None
        self.criterion = nn.HuberLoss()
        
        print(f"🚀 Using device: {self.device}")
    
    def load_all_gold_data(self):
        """Load ALL gold data directly from files for maximum data usage"""
        print("📊 Loading ALL historical gold data...")
        all_data = []
        
        # Check both parquet and CSV files
        gold_files = glob.glob("gold_data/gold_traffic_*.parquet") + glob.glob("gold_data/gold_traffic_*.csv")
        gold_files = [f for f in gold_files if not f.endswith(('_temp.parquet', '_temp.csv'))]
        
        if not gold_files:
            print("❌ No gold data files found")
            return pd.DataFrame()
        
        total_records = 0
        for file in sorted(gold_files):
            try:
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                
                # Ensure datetime format
                if 'Retrieval_Time' in df.columns:
                    df['Retrieval_Time'] = pd.to_datetime(df['Retrieval_Time'], errors='coerce')
                    df = df[df['Retrieval_Time'].notna()]
                
                record_count = len(df)
                total_records += record_count
                print(f"  📁 {os.path.basename(file)}: {record_count:,} records")
                all_data.append(df)
            except Exception as e:
                print(f"Error reading gold file {file}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"✅ Loaded ALL gold data: {len(combined):,} records from {len(gold_files)} files")
            return combined
        else:
            print("❌ No gold data could be loaded")
            return pd.DataFrame()
    
    def create_sample_data(self):
        """Create sample data for testing if gold data is not available"""
        print("📊 Creating sample data for testing...")
        
        # Generate realistic sample data
        np.random.seed(42)
        n_samples = 50000
        
        sample_data = {
            'LinkID': np.random.choice([f'LINK_{i:03d}' for i in range(1, 101)], n_samples),
            'RoadName': np.random.choice(['ORCHARD RD', 'SOMERSET RD', 'PATERSON RD', 'SIMS AVE', 'BUKIT TIMAH RD'], n_samples),
            'RoadCategory': np.random.choice(['1', '2', '3'], n_samples),
            'RoadCategory_Description': np.random.choice(['Expressways', 'Major Arterial Roads', 'Arterial Roads'], n_samples),
            'Retrieval_Time': pd.date_range('2024-01-01', periods=n_samples, freq='10min'),
            'AverageSpeed': np.random.uniform(10, 80, n_samples),
            'SpeedBand': np.random.randint(1, 8, n_samples),
            'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
            'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
            'day_sin': np.sin(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
            'day_cos': np.cos(2 * np.pi * np.random.randint(0, 7, n_samples) / 7),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'is_peak_morning': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'is_peak_evening': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'is_night': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'month': np.random.randint(1, 13, n_samples),
            'week_of_year': np.random.randint(1, 53, n_samples),
            'road_importance': np.random.randint(1, 6, n_samples),
            'optimal_speed': np.random.choice([70, 50, 40, 30, 60], n_samples),
            'speed_efficiency': np.random.uniform(0.3, 1.2, n_samples),
            'traffic_condition': np.random.choice(['Fluid', 'Moderate', 'Slow', 'Congested'], n_samples),
            'speed_rolling_1h': np.random.uniform(20, 70, n_samples),
            'speed_rolling_3h': np.random.uniform(25, 65, n_samples),
            'speed_volatility_1h': np.random.uniform(1, 15, n_samples),
            'speed_roc_1h': np.random.uniform(-0.3, 0.3, n_samples),
        }
        
        df = pd.DataFrame(sample_data)
        df['future_speed_1h'] = df['AverageSpeed'] + np.random.normal(0, 3, n_samples)
        
        return df
    
    def load_and_prepare_data(self, use_all_data=True):
        """Load gold data and prepare for training - ROBUST VERSION"""
        if use_all_data:
            print("📊 Loading ALL historical gold data for training...")
            gold_data = self.load_all_gold_data()
            
            if gold_data.empty:
                print("⚠️ No historical gold data available, trying recent data...")
                try:
                    gold_engineer = GoldFeatureEngineer()
                    gold_data = gold_engineer.get_latest_gold_data(hours=24*30)
                except Exception as e:
                    print(f"⚠️ Error loading recent gold data: {e}, using sample data")
                    gold_data = self.create_sample_data()
        else:
            print("📊 Loading recent gold data for training...")
            try:
                gold_engineer = GoldFeatureEngineer()
                gold_data = gold_engineer.get_latest_gold_data(hours=168)
            except Exception as e:
                print(f"⚠️ Error loading recent gold data: {e}, using sample data")
                gold_data = self.create_sample_data()
        
        if not gold_data.empty:
            print(f"✅ Loaded {len(gold_data):,} records with {len(gold_data.columns)} features")
            
            # Prepare target variable - predict next hour's average speed
            print("🔄 Preparing target variable (future_speed_1h)...")
            gold_data = gold_data.sort_values(['LinkID', 'Retrieval_Time'])
            gold_data['future_speed_1h'] = gold_data.groupby('LinkID')['AverageSpeed'].shift(-6)
            
            # Remove rows without future values
            initial_count = len(gold_data)
            gold_data = gold_data.dropna(subset=['future_speed_1h']).copy()
            target_removed = initial_count - len(gold_data)
            
            if target_removed > 0:
                print(f"  Removed {target_removed:,} records without future speed values")
        
        print(f"📈 Model data prepared: {len(gold_data):,} records")
        return gold_data
    
    def feature_engineering(self, df):
        """Prepare features for model training - ROBUST VERSION"""
        print("🔄 Engineering features for model...")
        
        # Select features for model
        feature_columns = [
            # Temporal features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_peak_morning', 'is_peak_evening', 'is_night',
            'month', 'week_of_year',
            
            # Traffic features
            'AverageSpeed', 'SpeedBand', 'road_importance', 'optimal_speed', 
            'speed_efficiency', 'traffic_condition',
            
            # Rolling features
            'speed_rolling_1h', 'speed_rolling_3h', 'speed_volatility_1h', 'speed_roc_1h',
            
            # Road category
            'RoadCategory'
        ]
        
        # Only use available columns
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features].copy()
        
        print(f"  Using {len(available_features)} features: {available_features}")
        
        # Handle categorical variables
        categorical_columns = ['traffic_condition', 'RoadCategory']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle NaN values in categorical columns
                    features_df[col] = features_df[col].fillna('Unknown')
                    features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    features_df[col] = features_df[col].fillna('Unknown')
                    # Handle unseen categories in validation/test data
                    known_categories = set(self.label_encoders[col].classes_)
                    current_categories = set(features_df[col].unique())
                    unseen_categories = current_categories - known_categories
                    
                    if unseen_categories:
                        print(f"  ⚠️ Found {len(unseen_categories)} unseen categories in {col}, mapping to 'Unknown'")
                        # Map unseen categories to a default value
                        features_df[col] = features_df[col].apply(
                            lambda x: x if x in known_categories else 'Unknown'
                        )
                    
                    features_df[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Target variable
        targets = df['future_speed_1h'].values
        
        print(f"✅ Feature engineering complete: {len(available_features)} features")
        return features_df.values, targets, available_features
    
    def prepare_datasets(self, features, targets):
        """Prepare train, validation, and test datasets - ROBUST VERSION"""
        print("📚 Preparing datasets...")
        
        # Handle case where we have very few samples
        if len(features) < 1000:
            print("⚠️ Very small dataset, adjusting splits...")
            self.config['validation_split'] = 0.1
            self.config['test_split'] = 0.1
            self.config['batch_size'] = 32
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, targets, 
            test_size=(self.config['validation_split'] + self.config['test_split']), 
            random_state=42,
            shuffle=True
        )
        
        # Calculate validation size relative to temp set
        val_ratio = self.config['validation_split'] / (self.config['validation_split'] + self.config['test_split'])
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=1-val_ratio,
            random_state=42
        )
        
        # Scale features
        print("  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create PyTorch datasets
        train_dataset = TrafficDataset(X_train_scaled, y_train)
        val_dataset = TrafficDataset(X_val_scaled, y_val)
        test_dataset = TrafficDataset(X_test_scaled, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        print(f"✅ Datasets prepared - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_features).squeeze()
            loss = self.criterion(predictions, batch_targets)
            loss.backward()
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features).squeeze()
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate additional metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        
        return total_loss / len(val_loader), mae, rmse, r2
    
    def train_model(self, train_loader, val_loader, input_size):
        """Main training loop with early stopping - ROBUST VERSION"""
        print("🎯 Starting model training...")
        
        # Initialize model
        self.model = TrafficPredictor(
            input_size=input_size,
            hidden_layers=self.config['hidden_layers'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # FIXED: Remove verbose parameter for compatibility
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_metrics_history = []
        self.previous_lr = self.config['learning_rate']
        
        print(f"📈 Training on {len(train_loader.dataset):,} samples, validating on {len(val_loader.dataset):,} samples")
        
        for epoch in range(self.config['epochs']):
            try:
                # Training
                train_loss = self.train_epoch(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                val_loss, mae, rmse, r2 = self.validate_epoch(val_loader)
                val_losses.append(val_loss)
                val_metrics_history.append({'mae': mae, 'rmse': rmse, 'r2': r2})
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # FIXED: Manual LR change detection
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr != self.previous_lr:
                    print(f'💡 Learning rate reduced to {current_lr:.2e}')
                    self.previous_lr = current_lr
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_model.pth')
                    print(f'✅ Epoch {epoch+1:03d}: New best model saved! Val Loss: {val_loss:.4f}, MAE: {mae:.2f}, R²: {r2:.4f}')
                else:
                    patience_counter += 1
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1:03d}/{self.config["epochs"]:03d} | '
                          f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                          f'MAE: {mae:.2f} | R²: {r2:.4f}')
                
                # Early stopping
                if patience_counter >= self.config['patience']:
                    print(f'🛑 Early stopping triggered after {epoch + 1} epochs')
                    break
                    
            except Exception as e:
                print(f"⚠️ Error in epoch {epoch+1}: {e}")
                # Continue training despite error in one epoch
                continue
        
        # Load best model safely
        try:
            self.load_model('best_model.pth')
            print("✅ Successfully loaded best model for final evaluation")
        except Exception as e:
            print(f"⚠️ Could not load best model: {e}")
            print("⚠️ Using current model state for evaluation")
        
        return train_losses, val_losses, val_metrics_history
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_features).squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        valid_targets = np.array(all_targets) != 0
        if np.any(valid_targets):
            mape = np.mean(np.abs((np.array(all_targets)[valid_targets] - np.array(all_predictions)[valid_targets]) / 
                                 np.array(all_targets)[valid_targets])) * 100
        else:
            mape = float('nan')
        
        print("🎯 Test Set Results:")
        print(f"   MAE:  {mae:.2f} km/h")
        print(f"   RMSE: {rmse:.2f} km/h")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²:   {r2:.4f}")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'metrics': {
                'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
                'mse': mse
            }
        }
    
    def save_model(self, filename):
        """Save model and metadata - FIXED for PyTorch 2.6+ compatibility"""
        model_path = os.path.join(self.model_dir, filename)
        
        # Save model state and metadata separately to avoid weights_only issues
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'feature_names': getattr(self, 'feature_names', []),
            'input_size': getattr(self, 'input_size', None),
            'saved_at': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Save model data (torch tensors only) with weights_only=True
        torch.save(model_data, model_path)
        
        # Save preprocessing objects separately using joblib (better for sklearn objects)
        preprocess_path = model_path.replace('.pth', '_preprocess.joblib')
        preprocess_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(preprocess_data, preprocess_path)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Preprocessing objects saved: {preprocess_path}")
    
    def load_model(self, filename):
        """Load model and metadata - FIXED for PyTorch 2.6+ compatibility"""
        model_path = os.path.join(self.model_dir, filename)
        preprocess_path = model_path.replace('.pth', '_preprocess.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data with weights_only=True for security
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"⚠️ Failed to load with weights_only=True: {e}")
            print("⚠️ Trying with weights_only=False (use with caution)")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load preprocessing objects separately
        if os.path.exists(preprocess_path):
            try:
                preprocess_data = joblib.load(preprocess_path)
                self.scaler = preprocess_data['scaler']
                self.label_encoders = preprocess_data['label_encoders']
            except Exception as e:
                print(f"⚠️ Error loading preprocessing objects: {e}")
                print("⚠️ Initializing new preprocessing objects")
                self.scaler = StandardScaler()
                self.label_encoders = {}
        else:
            print("⚠️ Preprocessing file not found, initializing new scaler and encoders")
            self.scaler = StandardScaler()
            self.label_encoders = {}
        
        # Reinitialize model if needed
        if self.model is None:
            input_size = checkpoint.get('input_size') or len(checkpoint.get('feature_names', []))
            if input_size:
                self.model = TrafficPredictor(
                    input_size=input_size,
                    hidden_layers=self.config['hidden_layers'],
                    dropout_rate=self.config['dropout_rate']
                ).to(self.device)
        
        if self.model:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_names = checkpoint.get('feature_names', [])
        self.input_size = checkpoint.get('input_size')
        
        print(f"📥 Model loaded: {model_path}")
    
    def plot_training_history(self, train_losses, val_losses, val_metrics):
        """Plot training history and metrics"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot losses
            ax1.plot(train_losses, label='Training Loss')
            ax1.plot(val_losses, label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot MAE
            mae_values = [m['mae'] for m in val_metrics]
            ax2.plot(mae_values, label='Validation MAE', color='orange')
            ax2.set_title('Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (km/h)')
            ax2.legend()
            ax2.grid(True)
            
            # Plot RMSE
            rmse_values = [m['rmse'] for m in val_metrics]
            ax3.plot(rmse_values, label='Validation RMSE', color='green')
            ax3.set_title('Validation RMSE')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('RMSE (km/h)')
            ax3.legend()
            ax3.grid(True)
            
            # Plot R²
            r2_values = [m['r2'] for m in val_metrics]
            ax4.plot(r2_values, label='Validation R²', color='red')
            ax4.set_title('Validation R² Score')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('R²')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Training history plot saved")
        except Exception as e:
            print(f"⚠️ Error creating training plots: {e}")
    
    def plot_predictions(self, test_results):
        """Plot prediction vs actual"""
        try:
            predictions = test_results['predictions']
            targets = test_results['targets']
            metrics = test_results['metrics']
            
            plt.figure(figsize=(12, 5))
            
            # Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(targets, predictions, alpha=0.6, s=10)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
            plt.xlabel('Actual Speed (km/h)')
            plt.ylabel('Predicted Speed (km/h)')
            plt.title(f'Predictions vs Actual\nR² = {metrics["r2"]:.4f}')
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(1, 2, 2)
            residuals = np.array(targets) - np.array(predictions)
            plt.scatter(predictions, residuals, alpha=0.6, s=10)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Speed (km/h)')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'predictions_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Predictions analysis plot saved")
        except Exception as e:
            print(f"⚠️ Error creating prediction plots: {e}")
    
    def run_full_pipeline(self):
        """Run the complete model development pipeline - ROBUST VERSION"""
        print("🚀 Starting Full Model Development Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load and prepare data - NOW WITH ALL HISTORICAL DATA
            model_data = self.load_and_prepare_data(use_all_data=True)
            
            if model_data.empty:
                print("❌ No data available for training")
                return None
                
            # 2. Feature engineering
            features, targets, feature_names = self.feature_engineering(model_data)
            self.feature_names = feature_names
            self.input_size = features.shape[1]
            
            # 3. Prepare datasets
            train_loader, val_loader, test_loader = self.prepare_datasets(features, targets)
            
            # 4. Train model
            train_losses, val_losses, val_metrics = self.train_model(
                train_loader, val_loader, input_size=features.shape[1]
            )
            
            # 5. Evaluate model
            test_results = self.evaluate_model(test_loader)
            
            # 6. Save final model and artifacts
            self.save_model('traffic_predictor_final.pth')
            
            # 7. Create visualizations
            self.plot_training_history(train_losses, val_losses, val_metrics)
            self.plot_predictions(test_results)
            
            # 8. Save training report
            self.save_training_report(test_results, len(model_data))
            
            print("🎉 Model development pipeline completed successfully!")
            print(f"📁 Model saved in: {os.path.abspath(self.model_dir)}")
            
            return test_results
            
        except Exception as e:
            print(f"❌ Error in model development pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_training_report(self, test_results, total_samples):
        """Save comprehensive training report"""
        try:
            report = {
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_samples': total_samples,
                    'device_used': str(self.device),
                    'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pytorch_version': torch.__version__
                },
                'model_config': self.config,
                'test_metrics': test_results['metrics'],
                'feature_names': self.feature_names,
                'model_architecture': {
                    'input_size': self.input_size,
                    'hidden_layers': self.config['hidden_layers'],
                    'dropout_rate': self.config['dropout_rate']
                }
            }
            
            report_path = os.path.join(self.model_dir, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📋 Training report saved: {report_path}")
        except Exception as e:
            print(f"⚠️ Error saving training report: {e}")

# Model Deployment Class - UPDATED for PyTorch 2.6+ compatibility
class TrafficModelDeployer:
    """Class for deploying and using the trained model"""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.input_size = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model for inference - ROBUST VERSION"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        preprocess_path = model_path.replace('.pth', '_preprocess.joblib')
        
        # Load model data with weights_only=True
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"⚠️ Failed to load with weights_only=True: {e}")
            print("⚠️ Trying with weights_only=False (use with caution)")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load preprocessing objects separately
        if os.path.exists(preprocess_path):
            try:
                preprocess_data = joblib.load(preprocess_path)
                self.scaler = preprocess_data['scaler']
                self.label_encoders = preprocess_data['label_encoders']
            except Exception as e:
                raise RuntimeError(f"Error loading preprocessing objects: {e}")
        else:
            raise FileNotFoundError(f"Preprocessing file not found: {preprocess_path}")
        
        # Recreate model architecture
        self.input_size = checkpoint.get('input_size') or len(checkpoint['feature_names'])
        self.model = TrafficPredictor(
            input_size=self.input_size,
            hidden_layers=checkpoint['config']['hidden_layers'],
            dropout_rate=checkpoint['config']['dropout_rate']
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_names = checkpoint['feature_names']
        
        print(f"📥 Model loaded successfully: {model_path}")
        print(f"🔧 Model configured for {len(self.feature_names)} features")
    
    def predict(self, features_df):
        """Make predictions on new data"""
        # Prepare features (same preprocessing as training)
        features_processed = self._preprocess_features(features_df)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_processed).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(features_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def _preprocess_features(self, features_df):
        """Preprocess features for inference"""
        # Select and order features
        available_features = [col for col in self.feature_names if col in features_df.columns]
        features_selected = features_df[available_features].copy()
        
        # Handle missing features
        missing_features = set(self.feature_names) - set(available_features)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            for feature in missing_features:
                features_selected[feature] = 0  # Fill with zeros
        
        # Handle categorical variables
        for col in self.label_encoders:
            if col in features_selected.columns:
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                current_categories = set(features_selected[col].unique())
                
                # Replace unseen categories with most frequent
                unseen_categories = current_categories - known_categories
                if unseen_categories:
                    most_frequent = self.label_encoders[col].classes_[0]
                    features_selected[col] = features_selected[col].apply(
                        lambda x: x if x in known_categories else most_frequent
                    )
                
                features_selected[col] = self.label_encoders[col].transform(features_selected[col])
        
        # Scale features
        features_scaled = self.scaler.transform(features_selected.values)
        
        return features_scaled
    
    def get_feature_importance(self):
        """Get feature importance from the first layer weights"""
        if self.model is None:
            return None
        
        # Get weights from first layer
        first_layer_weights = self.model.hidden_layers[0].weight.data.cpu().numpy()
        
        # Calculate feature importance as average absolute weight
        feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
        
        importance_dict = dict(zip(self.feature_names, feature_importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def main():
    """Main function to run model development"""
    print("🚀 Traffic Prediction Model Development")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer(model_dir="traffic_models")
    
    # Run full pipeline
    try:
        results = trainer.run_full_pipeline()
        
        if results:
            # Display final results
            metrics = results['metrics']
            print("\n🎯 FINAL MODEL PERFORMANCE:")
            print(f"   Mean Absolute Error: {metrics['mae']:.2f} km/h")
            print(f"   Root Mean Square Error: {metrics['rmse']:.2f} km/h")
            print(f"   Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
            print(f"   R² Score: {metrics['r2']:.4f}")
            
            print(f"\n💾 Model saved as: traffic_models/traffic_predictor_final.pth")
            print("📊 Training visualizations saved in traffic_models/")
            print("✅ Model development completed successfully!")
            
            # Show feature importance if available
            try:
                deployer = TrafficModelDeployer('traffic_models/traffic_predictor_final.pth')
                importance = deployer.get_feature_importance()
                if importance:
                    print("\n🔍 TOP 10 FEATURE IMPORTANCE:")
                    for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                        print(f"   {i+1:2d}. {feature}: {imp:.4f}")
            except Exception as e:
                print(f"⚠️ Could not compute feature importance: {e}")
                
        else:
            print("❌ Model development failed - no results returned")
        
    except Exception as e:
        print(f"❌ Model development failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()