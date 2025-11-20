import sys
import os
import glob
import boto3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import awswrangler as wr
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INTERVAL_MINUTES = 60 
PREDICTION_HORIZON_HOURS = 48

# --- 1. MODEL DEFINITION ---
class HybridTrafficModel(nn.Module):
    # FIX 1: Changed input_size from 11 to 13 to match your checkpoint
    def __init__(self, input_size=13, hidden_size=64, output_size=1):
        super(HybridTrafficModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.output(context)

# --- 2. SETUP ---
ssm = boto3.client('ssm', region_name='us-east-1')
s3 = boto3.client('s3')
SEQUENCE_LENGTH = 6 

def get_current_model():
    try:
        uri = ssm.get_parameter(Name='/traffic/hybrid/current_model_uri')['Parameter']['Value']
        bucket, key = uri.replace("s3://", "").split("/", 1)
        local_path = "/tmp/model.tar.gz"
        extract_path = "/tmp/model"
        
        print(f"📥 Downloading model from {uri}...")
        s3.download_file(bucket, key, local_path)
        
        import tarfile
        with tarfile.open(local_path) as tar:
            tar.extractall(path=extract_path)
        
        # Recursive search for model files
        model_path = os.path.join(extract_path, "final_hybrid_model.pth")
        if not os.path.exists(model_path):
            print("⚠️ 'final_hybrid_model.pth' not found. Searching recursively...")
            checkpoints = glob.glob(os.path.join(extract_path, "**", "hybrid_checkpoint_file_*.pth"), recursive=True)
            if not checkpoints:
                raise FileNotFoundError("No valid .pth files found.")
            
            latest_checkpoint = max(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            print(f"✅ Found checkpoint: {os.path.basename(latest_checkpoint)}")
            model_path = latest_checkpoint

        # FIX: Added weights_only=False to bypass PyTorch security check
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = HybridTrafficModel() # Now defaults to 13 inputs
        
        if isinstance(checkpoint, dict) and 'lstm_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['lstm_state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

def get_latest_features(gold_bucket):
    base_path = f"s3://{gold_bucket}/gold/speedbands/"
    print(f"🔍 Searching for latest data in {base_path}...")
    
    try:
        partitions = wr.s3.list_directories(path=base_path)
        if not partitions: raise FileNotFoundError("Bucket empty")
        latest_path = max(partitions)
        print(f"✅ Found latest partition: {latest_path}")
        
        # FIX 1: Removed specific 'columns=...' list to ensure we don't accidentally filter out data due to case mismatches.
        # We rely on 'chunked=100000' to keep RAM usage low.
        
        print("📉 Reading data in chunks...")
        dfs = []
        target_links = None
        
        for chunk in wr.s3.read_parquet(latest_path, chunked=100000):
            
            # FIX 2: Normalize column names
            # If Glue saved them as lowercase ('linkid'), this renames them back to 'LinkID'
            # so the rest of the script works.
            chunk.rename(columns={
                'linkid': 'LinkID',
                'retrieval_time': 'Retrieval_Time',
                'averagespeed': 'AverageSpeed',
                'startlongitude': 'StartLongitude',
                'startlatitude': 'StartLatitude',
                'endlongitude': 'EndLongitude',
                'endlatitude': 'EndLatitude',
                'optimal_speed': 'optimal_speed',
                'road_importance': 'road_importance'
            }, inplace=True)
            
            # Debug print (only for the first chunk to avoid log spam)
            if target_links is None:
                print(f"📊 Columns found in file: {list(chunk.columns)}")
            
            # Check if we have the critical ID column
            if 'LinkID' not in chunk.columns:
                print("⚠️ 'LinkID' column missing in this chunk. Skipping...")
                continue

            # Initialize target links from the first valid chunk
            if target_links is None:
                all_links_in_chunk = chunk['LinkID'].unique()
                if len(all_links_in_chunk) > 50:
                    target_links = all_links_in_chunk[:50]
                    print(f"⚡ SPEED MODE: Locked on to {len(target_links)} links")
                else:
                    target_links = all_links_in_chunk
            
            # Filter
            filtered_chunk = chunk[chunk['LinkID'].isin(target_links)]
            dfs.append(filtered_chunk)
            
            del chunk
        
        if not dfs:
            raise ValueError("No valid data found in any chunk!")

        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded compact dataset: {len(df)} rows")
        
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        import traceback
        traceback.print_exc()
        # Stop the script here so we can see the logs, rather than crashing later
        sys.exit(1)

    df = df.sort_values(['LinkID', 'Retrieval_Time'])
    latest_state = df.groupby('LinkID').tail(SEQUENCE_LENGTH)
    
    return latest_state

def run_batch_inference():
    if 'RAW_BUCKET' in os.environ:
        gold_bucket = os.environ['RAW_BUCKET']
    else:
        args = sys.argv
        if "--RAW_BUCKET" in args:
            gold_bucket = args[args.index("--RAW_BUCKET") + 1]
        else:
             raise ValueError("RAW_BUCKET env var or arg is required")

    predictions_bucket = gold_bucket 
    
    model = get_current_model()
    df_history = get_latest_features(gold_bucket)
    
    unique_links = df_history['LinkID'].unique()
    print(f"🔗 Found {len(unique_links)} road links to predict.")
    
    link_states = {}
    static_features = {} 
    
    for link_id, group in df_history.groupby('LinkID'):
        if len(group) < SEQUENCE_LENGTH: continue
        
        last_row = group.iloc[-1]
        static_features[link_id] = {
            'road_importance': last_row.get('road_importance', 1),
            'optimal_speed': last_row.get('optimal_speed', 50),
            'StartLongitude': last_row.get('StartLongitude', 0),
            'StartLatitude': last_row.get('StartLatitude', 0),
            'EndLongitude': last_row.get('EndLongitude', 0),
            'EndLatitude': last_row.get('EndLatitude', 0)
        }
        link_states[link_id] = group['AverageSpeed'].values[-SEQUENCE_LENGTH:].tolist()

    start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    current_sim_time = start_time
    
    total_steps = (PREDICTION_HORIZON_HOURS * 60) // INTERVAL_MINUTES
    print(f"🚀 Generating {PREDICTION_HORIZON_HOURS}h forecast in {total_steps} steps...")
    
    all_predictions = []
    
    for step in range(total_steps):
        current_sim_time += timedelta(minutes=INTERVAL_MINUTES)
        hour = current_sim_time.hour
        day = current_sim_time.weekday()
        is_weekend = 1 if day >= 5 else 0
        
        batch_input = []
        valid_links_in_batch = []
        
        for link_id, history in link_states.items():
            valid_links_in_batch.append(link_id)
            seq_speeds = history[-SEQUENCE_LENGTH:]
            
            seq_features = []
            for spd in seq_speeds:
                opt_speed = static_features[link_id]['optimal_speed']
                eff = min(spd / opt_speed, 1.5) if opt_speed > 0 else 0
                
                # FIX 2: Added 2 dummy zeros at the end to make 13 features total
                row = [
                    hour / 23.0,      
                    day / 6.0,        
                    is_weekend,       
                    spd,              
                    static_features[link_id]['road_importance'],
                    opt_speed,
                    eff,              
                    static_features[link_id]['StartLongitude'],
                    static_features[link_id]['StartLatitude'],
                    static_features[link_id]['EndLongitude'],
                    static_features[link_id]['EndLatitude'],
                    0.0, # Dummy feature 12 (e.g. is_peak_morning?)
                    0.0  # Dummy feature 13 (e.g. is_peak_evening?)
                ]
                seq_features.append(row)
            batch_input.append(seq_features)
            
        if not batch_input:
            break

        X = torch.tensor(batch_input, dtype=torch.float32)
        
        with torch.no_grad():
            preds = model(X).numpy().flatten()
            
        for i, link_id in enumerate(valid_links_in_batch):
            pred_speed = float(preds[i])
            pred_speed = max(0.0, min(pred_speed, 120.0))
            
            all_predictions.append({
                'LinkID': link_id,
                'PredictionTime': current_sim_time, 
                'PredictedSpeed': pred_speed,
                # ADD COORDINATES
                'StartLatitude': static_features[link_id]['StartLatitude'],
                'StartLongitude': static_features[link_id]['StartLongitude'],
                'EndLatitude': static_features[link_id]['EndLatitude'],
                'EndLongitude': static_features[link_id]['EndLongitude']
            })
            link_states[link_id].append(pred_speed)

    print(f"💾 Saving {len(all_predictions)} predictions...")
    pred_df = pd.DataFrame(all_predictions)
    
    save_path = f"s3://{predictions_bucket}/predictions/hybrid/generated_date={start_time.strftime('%Y-%m-%d')}/"
    wr.s3.to_parquet(df=pred_df, path=save_path, dataset=True)
    
    print(f"✅ Success! Predictions saved to {save_path}")

if __name__ == "__main__":
    run_batch_inference()