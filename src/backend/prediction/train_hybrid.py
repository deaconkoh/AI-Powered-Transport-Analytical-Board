import os
import subprocess
import sys
from datetime import datetime
import boto3

def ensure_xgboost():
    try:
        import xgboost  # noqa: F401
        print("✅ xgboost already installed")
    except ModuleNotFoundError:
        print("⚠️ xgboost not found, installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        print("✅ xgboost installed")

ensure_xgboost()

from backend.prediction import cloud_model
from backend.prediction.cloud_model import HybridTrainer

REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_GOLD_LOCAL_DIR = os.environ.get("GOLD_DATA_PATH", "/opt/ml/input/gold_data")
GOLD_DATA_S3_PREFIX = os.environ.get("GOLD_DATA_S3_PREFIX")
MODEL_SAVE_DIR = os.environ.get("SM_MODEL_DIR", "./traffic_models_hybrid")


def download_gold_from_s3(local_dir: str) -> str:
    """
    If GOLD_DATA_S3_PREFIX is set (s3://bucket/prefix),
    download all .parquet files under that prefix into local_dir.
    Otherwise, just make sure local_dir exists and return it.
    """
    if not GOLD_DATA_S3_PREFIX:
        print("GOLD_DATA_S3_PREFIX not set – assuming gold data already exists at", local_dir)
        os.makedirs(local_dir, exist_ok=True)
        return local_dir

    print(f"📥 Downloading gold data from {GOLD_DATA_S3_PREFIX} to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION)
    prefix_str = GOLD_DATA_S3_PREFIX.replace("s3://", "")
    if "/" in prefix_str:
        bucket, prefix = prefix_str.split("/", 1)
    else:
        bucket, prefix = prefix_str, ""

    paginator = s3.get_paginator("list_objects_v2")
    total_files = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            print(f"  🔽 {key} → {local_path}")
            s3.download_file(bucket, key, local_path)
            total_files += 1

    print(f"✅ Downloaded {total_files} parquet file(s) into {local_dir}")
    return local_dir


def main():
    
    local_gold_dir = download_gold_from_s3(DEFAULT_GOLD_LOCAL_DIR)

    cloud_model.GOLD_DATA_PATH = local_gold_dir
    cloud_model.MODEL_SAVE_PATH = MODEL_SAVE_DIR
    os.makedirs(cloud_model.MODEL_SAVE_PATH, exist_ok=True)

    print("🔧 Configured paths:")
    print("   GOLD_DATA_PATH  =", cloud_model.GOLD_DATA_PATH)
    print("   MODEL_SAVE_PATH =", cloud_model.MODEL_SAVE_PATH)

    start_time = datetime.now()
    try:
        print("🚀 Starting Hybrid training...")
        trainer = HybridTrainer()
        trainer.run_hybrid_training(max_files=1)

        total_time = datetime.now() - start_time
        print(f"\n✅ Training completed in {total_time}")
        print(f"💾 Models saved under: {cloud_model.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
