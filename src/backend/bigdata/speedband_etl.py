# src/bigdata/speedband_etl.py

import os
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

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


def read_bronze(hours=24):
    """
    Simple version: read all Bronze snapshots under the prefix.
    In the report, you can mention filtering to last N hours via partitioning.
    """
    path = f"{BRONZE_PREFIX}*.json"
    df_raw = spark.read.json(path)
    return df_raw


def to_silver(df_raw):
    # Explode 'value' array: one row per speedband record
    exploded = (
        df_raw
        .select(
            F.to_timestamp("fetched_at", "yyyyMMdd-HHmmss").alias("retrieval_time"),
            F.explode("value").alias("rec")
        )
    )

    df = exploded.select(
        "retrieval_time",
        F.col("rec.LinkID").alias("link_id"),
        F.col("rec.MinimumSpeed").cast("double").alias("MinimumSpeed"),
        F.col("rec.MaximumSpeed").cast("double").alias("MaximumSpeed"),
        F.col("rec.StartLon").cast("double").alias("StartLongitude"),
        F.col("rec.StartLat").cast("double").alias("StartLatitude"),
        F.col("rec.EndLon").cast("double").alias("EndLongitude"),
        F.col("rec.EndLat").cast("double").alias("EndLatitude"),
    )

    # Deduplicate
    df = df.dropDuplicates(["link_id", "retrieval_time"])

    # Filter invalid timestamps
    df = df.filter(df.retrieval_time <= F.current_timestamp())

    # Speed bounds
    df = df.filter((df.MinimumSpeed >= 0) & (df.MinimumSpeed <= 150))
    df = df.filter((df.MaximumSpeed >= 0) & (df.MaximumSpeed <= 150))

    # Average speed
    df = df.withColumn(
        "AverageSpeed",
        (F.col("MinimumSpeed") + F.col("MaximumSpeed")) / 2.0
    ).filter((F.col("AverageSpeed") >= 0) & (F.col("AverageSpeed") <= 120))

    # Rough Singapore bounds
    df = df.filter(
        (F.col("StartLongitude") >= 103.6) & (F.col("StartLongitude") <= 104.1) &
        (F.col("StartLatitude")  >= 1.2)   & (F.col("StartLatitude")  <= 1.5)
    )

    # Partition columns
    df = df.withColumn("date", F.to_date("retrieval_time"))
    df = df.withColumn("hour", F.hour("retrieval_time"))

    return df


def write_silver(df_silver):
    (
        df_silver.write
        .mode("append")
        .partitionBy("date", "hour")
        .parquet(SILVER_PREFIX)
    )


def to_gold(df_silver):
    df = df_silver

    # Time features
    df = df.withColumn("hour_of_day", F.hour("retrieval_time"))
    df = df.withColumn("day_of_week", F.dayofweek("retrieval_time"))
    df = df.withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))

    # Rolling window (last 60 minutes per link)
    w_60 = (
        Window
        .partitionBy("link_id")
        .orderBy(F.col("retrieval_time").cast("long"))
        .rangeBetween(-60 * 60, 0)
    )

    df = df.withColumn("avg_speed_60m", F.avg("AverageSpeed").over(w_60))
    df = df.withColumn("vol_speed_60m", F.stddev_pop("AverageSpeed").over(w_60))

    return df


def write_gold(df_gold):
    (
        df_gold.write
        .mode("append")
        .partitionBy("date", "hour")
        .parquet(GOLD_PREFIX)
    )


def run_etl():
    df_bronze = read_bronze(hours=24)
    if df_bronze.rdd.isEmpty():
        print("⚠️ No Bronze data to process")
        return

    df_silver = to_silver(df_bronze)
    write_silver(df_silver)
    print("✅ Silver written")

    df_gold = to_gold(df_silver)
    write_gold(df_gold)
    print("✅ Gold written")


if __name__ == "__main__":
    run_etl()
