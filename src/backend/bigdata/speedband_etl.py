import os
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import boto3  

# --- Glue args / local fallback ---
try:
    from awsglue.utils import getResolvedOptions
    args = getResolvedOptions(sys.argv, ['RAW_BUCKET'])
    RAW_BUCKET = args['RAW_BUCKET']
except Exception:
    # Local dev: set RAW_BUCKET via env
    RAW_BUCKET = os.environ.get("RAW_BUCKET")
    if not RAW_BUCKET:
        raise RuntimeError("RAW_BUCKET not set (Glue arg or env var required)")

# S3 paths
BRONZE_PREFIX = f"s3://{RAW_BUCKET}/raw/lta/speedbands/"
SILVER_PREFIX = f"s3://{RAW_BUCKET}/silver/speedbands/"
GOLD_PREFIX   = f"s3://{RAW_BUCKET}/gold/speedbands/"

spark = SparkSession.builder.appName("SpeedbandETL").getOrCreate()

def get_latest_date_from_s3():
    """Helper: Scans S3 to find the latest YYYYMMDD prefix available."""
    s3 = boto3.client('s3')
    # Extract bucket and prefix from BRONZE_PREFIX (s3://bucket/prefix/)
    # We know BRONZE_PREFIX is f"s3://{RAW_BUCKET}/raw/lta/speedbands/"
    prefix = "raw/lta/speedbands/"
    
    try:
        latest_date = ""
        paginator = s3.get_paginator('list_objects_v2')
        # We list the bucket to find files
        for page in paginator.paginate(Bucket=RAW_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Key format example: raw/lta/speedbands/20251120-120000.json
                    # We want to extract '20251120'
                    filename = obj['Key'].split('/')[-1]
                    if '-' in filename:
                        date_part = filename.split('-')[0]
                        # Simple string comparison works for ISO dates (YYYYMMDD)
                        if date_part > latest_date:
                            latest_date = date_part
                            
        return latest_date if latest_date else None
    except Exception as e:
        print(f"⚠️ Warning: Failed to list S3 for fallback: {e}")
        return None

def read_bronze_for_date(process_date=None):
    """
    Smart Daily Read:
    1. Try to read 'Yesterday' (Standard Daily ETL).
    2. If missing/empty, fallback to 'Latest Available Date' (Resilience).
    """
    if process_date is None:
        # Default: Process yesterday's data
        process_date = (datetime.utcnow() - timedelta(days=1)).date()

    target_date_str = process_date.strftime("%Y%m%d")
    
    # Target path: specific date files only
    path = f"{BRONZE_PREFIX}{target_date_str}*.json"
    print(f"📖 Primary Strategy: Reading data for {target_date_str}...")
    
    try:
        df = spark.read.json(path)
        
        if df.rdd.isEmpty():
            raise ValueError("Dataframe is empty")
            
        print(f"✅ Success: Found data for {target_date_str}")
        return df
        
    except Exception as e:
        print(f"⚠️ Primary failed: Data for {target_date_str} unavailable. Reason: {e}")
        print("🔄 Fallback Strategy: Searching for latest available data...")
        
        latest_date = get_latest_date_from_s3()
        
        if latest_date:
            print(f"✅ Found latest available data: {latest_date}")
            fallback_path = f"{BRONZE_PREFIX}{latest_date}*.json"
            return spark.read.json(fallback_path)
        else:
            print("❌ Critical: No data found in bucket at all.")
            # Return empty schema to prevent crash
            return spark.createDataFrame([], schema="value STRING, fetched_at STRING")

# ---------- SILVER: match friend’s silver_tier schema / logic ----------

def to_silver(df_raw):
    """
    Transform Bronze JSON into Silver parquet in a way that matches the
    columns used by `silver_tier.py` and `gold_tier.py`.

    Key columns we keep:
    - Retrieval_Time
    - LinkID, RoadName, RoadCategory, RoadCategory_Description
    - MinimumSpeed, MaximumSpeed, AverageSpeed
    - StartLongitude, StartLatitude, EndLongitude, EndLatitude
    - SpeedBand
    """

    exploded = (
        df_raw
        .select(
            F.to_timestamp("fetched_at", "yyyyMMdd-HHmmss").alias("Retrieval_Time"),
            F.explode("value").alias("rec")
        )
    )

    # 2. SAFE COLUMN SELECTION
    # Check if 'RoadCategory_Description' exists in the schema inferred from JSON
    rec_fields = exploded.schema["rec"].dataType.names
    
    if "RoadCategory_Description" in rec_fields:
        rc_desc_col = F.col("rec.RoadCategory_Description")
    else:
        print("⚠️ 'RoadCategory_Description' missing in source data. Filling with NULL.")
        rc_desc_col = F.lit(None).cast("string")

    # 3. Select columns safely
    df = exploded.select(
        "Retrieval_Time",
        F.col("rec.LinkID").alias("LinkID"),
        F.col("rec.RoadName").alias("RoadName"),
        F.col("rec.RoadCategory").alias("RoadCategory"),
        rc_desc_col.alias("RoadCategory_Description"), # <--- Uses the safe version
        F.col("rec.SpeedBand").alias("SpeedBand"),
        F.col("rec.MinimumSpeed").cast("double").alias("MinimumSpeed"),
        F.col("rec.MaximumSpeed").cast("double").alias("MaximumSpeed"),
        F.col("rec.StartLon").cast("double").alias("StartLongitude"),
        F.col("rec.StartLat").cast("double").alias("StartLatitude"),
        F.col("rec.EndLon").cast("double").alias("EndLongitude"),
        F.col("rec.EndLat").cast("double").alias("EndLatitude"),
    )

    # Remove duplicates
    df = df.dropDuplicates()

    # Filter invalid timestamps
    df = df.filter(df.Retrieval_Time.isNotNull())
    df = df.filter(df.Retrieval_Time <= F.current_timestamp())

    # Numeric sanity filters
    df = df.filter((F.col("MinimumSpeed") >= 0) & (F.col("MinimumSpeed") <= 150))
    df = df.filter((F.col("MaximumSpeed") >= 0) & (F.col("MaximumSpeed") <= 150))

    # AverageSpeed with clipping
    df = df.withColumn(
        "AverageSpeed",
        (F.coalesce(F.col("MinimumSpeed"), F.lit(0.0)) +
         F.coalesce(F.col("MaximumSpeed"), F.lit(0.0))) / 2.0
    )

    df = df.withColumn(
        "AverageSpeed",
        F.when(F.col("AverageSpeed") < 0, 0.0)
         .when(F.col("AverageSpeed") > 120, 120.0)
         .otherwise(F.col("AverageSpeed"))
    )

    # Rough Singapore bounds
    df = df.filter(
        (F.col("StartLongitude").between(103.5, 104.3)) &
        (F.col("StartLatitude").between(1.1, 1.6))
    )

    # Partition columns
    df = df.withColumn("date", F.to_date("Retrieval_Time"))
    df = df.withColumn("hour", F.hour("Retrieval_Time"))

    return df

def write_silver(df_silver):
    (
        df_silver.write
        .mode("append")
        .partitionBy("date")
        .parquet(SILVER_PREFIX)
    )


# ---------- GOLD: replicate gold_tier.py feature engineering in Spark ----------

def to_gold(df_silver):
    """
    Create Gold features equivalent to GoldFeatureEngineer in gold_tier.py
    using PySpark.

    Time features:
      - hour, day_of_week (Mon=0..Sun=6), month
      - hour_sin, hour_cos (cyclical encoding)
      - is_weekend, is_peak_morning, is_peak_evening

    Traffic features:
      - road_importance, optimal_speed, speed_efficiency, traffic_condition

    Final columns match friend's `essential_columns`.
    """

    df = df_silver

    # Ensure Retrieval_Time is timestamp
    df = df.withColumn("Retrieval_Time", F.to_timestamp("Retrieval_Time"))
    df = df.filter(F.col("Retrieval_Time").isNotNull())

    # ----- Time features -----
    df = df.withColumn("hour", F.hour("Retrieval_Time"))
    df = df.withColumn(
        "day_of_week",
        (F.dayofweek("Retrieval_Time") + 5) % 7
    )
    df = df.withColumn("month", F.month("Retrieval_Time"))

    # Cyclical encoding
    two_pi = 2.0 * 3.141592653589793 / 24.0
    df = df.withColumn("hour_sin", F.sin(F.col("hour") * F.lit(two_pi)))
    df = df.withColumn("hour_cos", F.cos(F.col("hour") * F.lit(two_pi)))

    # Flags
    df = df.withColumn("is_weekend", (F.col("day_of_week") >= 5).cast("int"))
    df = df.withColumn("is_peak_morning", ((F.col("hour") >= 7) & (F.col("hour") <= 9)).cast("int"))
    df = df.withColumn("is_peak_evening", ((F.col("hour") >= 17) & (F.col("hour") <= 19)).cast("int"))

    # ----- Traffic features (SAFE MODE) -----
    # Check if the description column exists. If not, we cannot do the mapping.
    has_desc = "RoadCategory_Description" in df.columns

    if has_desc:
        # 1. Road Importance
        ri_map = {
            'Expressways': 5, 'Major Arterial Roads': 4, 'Arterial Roads': 3,
            'Minor Arterial Roads': 2, 'Small Roads': 1, 'Slip Roads': 1, 'Short Tunnels': 3
        }
        ri_pairs = []
        for k, v in ri_map.items(): ri_pairs.extend([F.lit(k), F.lit(v)])
        ri_map_col = F.create_map(*ri_pairs)

        df = df.withColumn("road_importance", 
            F.when(F.col("RoadCategory_Description").isNotNull(), ri_map_col.getItem(F.col("RoadCategory_Description")))
             .otherwise(F.lit(1))
        )

        # 2. Optimal Speed
        os_map = {
            'Expressways': 70, 'Major Arterial Roads': 50, 'Arterial Roads': 40,
            'Minor Arterial Roads': 30, 'Small Roads': 20, 'Slip Roads': 25, 'Short Tunnels': 40
        }
        os_pairs = []
        for k, v in os_map.items(): os_pairs.extend([F.lit(k), F.lit(v)])
        os_map_col = F.create_map(*os_pairs)

        df = df.withColumn("optimal_speed", 
            F.when(F.col("RoadCategory_Description").isNotNull(), os_map_col.getItem(F.col("RoadCategory_Description")))
             .otherwise(F.lit(30.0))
        )
    else:
        print("⚠️ 'RoadCategory_Description' missing in Silver. Using defaults for Gold.")
        df = df.withColumn("road_importance", F.lit(1))
        df = df.withColumn("optimal_speed", F.lit(30.0))

    # Speed efficiency
    df = df.withColumn("speed_efficiency", F.col("AverageSpeed") / F.col("optimal_speed"))
    df = df.withColumn("speed_efficiency", 
        F.when(F.col("speed_efficiency") < 0, 0.0)
         .when(F.col("speed_efficiency") > 1.5, 1.5)
         .otherwise(F.col("speed_efficiency"))
    )

    # Traffic condition
    df = df.withColumn("traffic_condition",
        F.when(F.col("AverageSpeed") >= 50, F.lit("Fluid"))
         .when(F.col("AverageSpeed") >= 30, F.lit("Moderate"))
         .when(F.col("AverageSpeed") >= 15, F.lit("Slow"))
         .otherwise(F.lit("Congested"))
    )

    # Select final columns
    essential_columns = [
        'LinkID', 'RoadName', 'RoadCategory', 'RoadCategory_Description',
        'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
        'is_weekend', 'is_peak_morning', 'is_peak_evening',
        'AverageSpeed', 'SpeedBand', 'road_importance', 'optimal_speed',
        'speed_efficiency', 'traffic_condition',
        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
        'Retrieval_Time'
    ]
    
    # Safety: Only select columns that actually exist
    available_columns = [c for c in essential_columns if c in df.columns]
    df = df.select(*available_columns)

    # Drop missing critical data
    if 'AverageSpeed' in df.columns:
        df = df.dropna(subset=['AverageSpeed'])

    # Add partitioning dates
    if "Retrieval_Time" in df.columns:
        df = df.withColumn("date", F.to_date("Retrieval_Time"))

    return df


def write_gold(df_gold):
    (
        df_gold.write
        .mode("append")
        .partitionBy("date")
        .parquet(GOLD_PREFIX)
    )


def run_etl():
    df_bronze = read_bronze_for_date()

    if df_bronze.rdd.isEmpty():
        print("⚠️ No Bronze data to process")
        return

    # 1. Create Silver
    df_silver = to_silver(df_bronze)
    df_silver.cache()
    print(f"📊 Silver Row Count: {df_silver.count()}") 
    # ----------------------------------

    write_silver(df_silver)
    print("✅ Silver written")

    # 2. Create Gold (Reads from Cached Silver)
    df_gold = to_gold(df_silver)
    write_gold(df_gold)
    print("✅ Gold written")

    # Clean up memory
    df_silver.unpersist()


if __name__ == "__main__":
    run_etl()
