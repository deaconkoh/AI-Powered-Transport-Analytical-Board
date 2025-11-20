import boto3
import os
import pandas as pd
import awswrangler as wr
from datetime import datetime, timedelta

# Config
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'traffic-ai')
ROLE_ARN = os.environ['ROLE_ARN']
RAW_BUCKET = os.environ['RAW_BUCKET']
MODELS_BUCKET = os.environ['MODELS_BUCKET']
METRIC_PARAM = "/traffic/hybrid/current_metric"

sm = boto3.client('sagemaker')
ssm = boto3.client('ssm')

def get_yesterday_mae():
    """
    Calculates real MAE by comparing Gold Data vs. Batch Predictions
    """
    # 1. Define Dates
    # We check "Yesterday" because that's the most recent full day of data we have
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📊 Calculating performance for date: {yesterday}")

    try:
        # 2. Read Actuals (Gold)
        # Path: s3://bucket/gold/speedbands/date=YYYY-MM-DD/
        actuals_path = f"s3://{RAW_BUCKET}/gold/speedbands/date={yesterday}/"
        
        # Only read columns we need to save RAM
        df_actual = wr.s3.read_parquet(
            path=actuals_path,
            columns=['LinkID', 'Retrieval_Time', 'AverageSpeed']
        )
        
        # Round Actuals to nearest hour to match predictions (since batch pred is hourly)
        # Note: If your predictions are 15 mins, round to 15 mins.
        df_actual['JoinTime'] = pd.to_datetime(df_actual['Retrieval_Time']).dt.round('H')
        
        # Aggregate actuals (if multiple records per hour, take average)
        df_actual = df_actual.groupby(['LinkID', 'JoinTime'])['AverageSpeed'].mean().reset_index()

        # 3. Read Predictions
        # We look for predictions generated yesterday (which cover yesterday + today)
        # Path: s3://bucket/predictions/hybrid/generated_date=YYYY-MM-DD/
        preds_path = f"s3://{RAW_BUCKET}/predictions/hybrid/generated_date={yesterday}/"
        
        df_pred = wr.s3.read_parquet(
            path=preds_path,
            columns=['LinkID', 'PredictionTime', 'PredictedSpeed']
        )
        df_pred['JoinTime'] = pd.to_datetime(df_pred['PredictionTime']).dt.round('H')

        # 4. Merge (Inner Join)
        # We only evaluate where we have BOTH a prediction and an actual value
        df_merged = pd.merge(
            df_actual, 
            df_pred, 
            on=['LinkID', 'JoinTime'], 
            suffixes=('_actual', '_pred')
        )
        
        if len(df_merged) == 0:
            print("⚠️ No overlapping data found between Actuals and Predictions.")
            return None

        # 5. Calculate MAE
        # Abs(Actual - Pred)
        df_merged['error'] = abs(df_merged['AverageSpeed'] - df_merged['PredictedSpeed'])
        mae = df_merged['error'].mean()
        
        print(f"✅ Calculated MAE based on {len(df_merged)} data points: {mae:.4f}")
        return mae

    except Exception as e:
        print(f"⚠️ Could not calculate real MAE: {e}")
        return None

def lambda_handler(event, context):
    print("🔍 Starting Model Health Check...")
    
    # --- TEST MODE BLOCK ---
    # If we manually trigger the Lambda with {"force_retrain": true},
    # we bypass the data check and pretend the model is broken.
    if event.get("force_retrain", False):
        print("🧪 TEST MODE DETECTED: Simulating model failure...")
        real_mae = 999.9 # Fake high error
    else:
        # Normal operation: Check real data
        real_mae = get_yesterday_mae()
    # -----------------------

    # 1. Get Baseline
    try:
        baseline_mae = float(ssm.get_parameter(Name=METRIC_PARAM)['Parameter']['Value'])
    except:
        baseline_mae = 10.0 
    
    print(f"📏 Baseline MAE: {baseline_mae}")

    # Handle the case where data is missing (and we aren't testing)
    if real_mae is None:
        print("⚠️ Skipping check (insufficient data).")
        return {"status": "skipped"}

    # 3. Degradation Logic
    DRIFT_THRESHOLD_RATIO = 1.2
    ABSOLUTE_THRESHOLD = 15.0
    
    is_degraded = (real_mae > (baseline_mae * DRIFT_THRESHOLD_RATIO)) or (real_mae > ABSOLUTE_THRESHOLD)
    
    if is_degraded:
        print(f"🚨 ALERT: Model Degraded! Real MAE ({real_mae:.2f}) is too high compared to Baseline ({baseline_mae:.2f}).")
        print("🚀 Triggering Auto-Retraining...")
        
        job_name = f"{PROJECT_NAME}-auto-retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 4. Trigger Training
        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04',
                'TrainingInputMode': 'File',
            },
            RoleArn=ROLE_ARN,
            OutputDataConfig={
                'S3OutputPath': f"s3://{os.environ['MODELS_BUCKET']}/hybrid/training-output"
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={'MaxRuntimeInSeconds': 86400},
            HyperParameters={
                'sagemaker_program': 'backend/prediction/train_hybrid.py',
                'sagemaker_submit_directory': f"s3://{os.environ['MODELS_BUCKET']}/code/hybrid-training.tar.gz",
                'GOLD_DATA_S3_PREFIX': f"s3://{RAW_BUCKET}/gold/speedbands/"
            },
            EnableNetworkIsolation=False
        )
        return {"status": "retraining_triggered", "new_mae": real_mae, "job": job_name}
        
    else:
        print(f"✅ Model is healthy. Real MAE ({real_mae:.2f}) is within acceptable limits.")
        return {"status": "healthy", "current_mae": real_mae}