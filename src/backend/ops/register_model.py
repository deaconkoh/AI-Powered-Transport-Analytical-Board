import boto3
import json
import tarfile
import os
import io
import numpy as np
from datetime import datetime

s3 = boto3.client('s3')
ssm = boto3.client('ssm')
sm = boto3.client('sagemaker')

# CONFIG
MODELS_BUCKET = os.environ['MODELS_BUCKET']
SSM_PARAM_URI = "/traffic/hybrid/current_model_uri"
SSM_PARAM_METRIC = "/traffic/hybrid/current_metric" 

def get_current_best_metric():
    """Fetches the metric (MAE) of the currently deployed model."""
    try:
        response = ssm.get_parameter(Name=SSM_PARAM_METRIC)
        return float(response['Parameter']['Value'])
    except ssm.exceptions.ParameterNotFound:
        print("⚠️ No current metric found. Assuming first run (infinite error).")
        return float('inf')

def update_ssm(param_name, value):
    ssm.put_parameter(
        Name=param_name,
        Value=value,
        Type='String',
        Overwrite=True
    )

def lambda_handler(event, context):
    print("Event:", json.dumps(event))
    
    # 1. Parse EventBridge details
    job_name = event['detail']['TrainingJobName']
    
    # 2. Find the S3 artifact location
    job_desc = sm.describe_training_job(TrainingJobName=job_name)
    model_artifact_url = job_desc['ModelArtifacts']['S3ModelArtifacts']
    
    # Parse bucket and key (e.g., s3://bucket/key)
    parts = model_artifact_url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    
    print(f"📥 Inspecting artifact: {bucket}/{key}")

    # 3. Download and extract report from model.tar.gz in memory
    obj = s3.get_object(Bucket=bucket, Key=key)
    bytestream = io.BytesIO(obj['Body'].read())
    
    report_data = {}
    
    with tarfile.open(fileobj=bytestream, mode="r:gz") as tar:
        try:
            report_file = tar.extractfile("hybrid_training_report.json")
            report_data = json.load(report_file)
        except KeyError:
            print("❌ hybrid_training_report.json not found in tarball! Aborting.")
            return {"status": "failed", "reason": "no_report"}
            
    # 4. Calculate Metric (Average XGBoost MAE from file history)
    # Your cloud_model.py saves a list of file progress. We average the xgb_loss.
    file_progress = report_data.get('file_progress', [])
    xgb_losses = [entry['xgb_loss'] for entry in file_progress if entry['xgb_loss'] > 0]
    
    if not xgb_losses:
        print("❌ No valid XGBoost losses found in report.")
        new_mae = float('inf')
    else:
        new_mae = np.mean(xgb_losses)
        
    print(f"📊 New Model MAE: {new_mae}")
    
    # 5. Archive to Versioned Folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_prefix = f"hybrid/v{timestamp}"
    
    # Copy the artifact to a "permanent" versioned home
    new_artifact_key = f"{version_prefix}/model.tar.gz"
    s3.copy_object(
        Bucket=MODELS_BUCKET,
        CopySource={'Bucket': bucket, 'Key': key},
        Key=new_artifact_key
    )
    
    # Save a cleaned metadata file for easier debugging later
    metadata = {
        "job_name": job_name,
        "version_id": timestamp,
        "mae": new_mae,
        "original_report": report_data,
        "artifact_uri": f"s3://{MODELS_BUCKET}/{new_artifact_key}"
    }
    s3.put_object(
        Bucket=MODELS_BUCKET,
        Key=f"{version_prefix}/metadata.json",
        Body=json.dumps(metadata)
    )
    
    # 6. Evaluation & Promotion
    current_best = get_current_best_metric()
    
    if new_mae < current_best:
        print(f"✅ PROMOTING: New MAE {new_mae:.4f} < Current {current_best:.4f}")
        # Update the pointer to the NEW versioned artifact
        update_ssm(SSM_PARAM_URI, f"s3://{MODELS_BUCKET}/{new_artifact_key}")
        # Update the "high score"
        update_ssm(SSM_PARAM_METRIC, str(new_mae))
    else:
        print(f"🛑 REJECTING: New MAE {new_mae:.4f} >= Current {current_best:.4f}")
        # We do nothing. The SSM pointer stays on the old, better model.
        # This effectively handles your "rollback" logic—bad models never go live.

    return {"status": "success", "promoted": new_mae < current_best}