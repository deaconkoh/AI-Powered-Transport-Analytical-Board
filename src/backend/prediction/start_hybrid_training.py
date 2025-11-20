# start_hybrid_training.py
import os
from datetime import datetime
import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")

SM_CLIENT = boto3.client("sagemaker", region_name=REGION)

# === CONFIG ===
MODELS_BUCKET = "traffic-ai-models-754029048130-us-east-1"
PROJECT_NAME  = "traffic-ai"
RAW_BUCKET    = "traffic-ai-raw-754029048130-us-east-1"

TRAINING_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
CODE_S3_URI    = f"s3://{MODELS_BUCKET}/code/hybrid-training.tar.gz"
OUTPUT_S3_PREFIX = f"s3://{MODELS_BUCKET}/hybrid/training-output/"



def start_training_job():
    job_name = f"{PROJECT_NAME}-hybrid-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    print(f"🚀 Starting SageMaker training job: {job_name}")

    response = SM_CLIENT.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn="arn:aws:iam::754029048130:role/traffic-ai-sm-exec-role",  # aws_iam_role.sm_exec.arn
        OutputDataConfig={
            "S3OutputPath": OUTPUT_S3_PREFIX,
        },
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 3600,
        },
        HyperParameters = {
            "sagemaker_program":          "backend/prediction/train_hybrid.py",
            "sagemaker_submit_directory": CODE_S3_URI,
        },
        Environment = {
            "AWS_REGION":          REGION,
            "GOLD_DATA_S3_PREFIX": f"s3://{RAW_BUCKET}/gold/speedbands/",
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{RAW_BUCKET}/gold/speedbands/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
                "InputMode": "File",
            }
        ],

    )

    print("✅ create_training_job response:")
    print(response)


if __name__ == "__main__":
    start_training_job()
