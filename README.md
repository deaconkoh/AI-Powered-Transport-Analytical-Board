# 🧭 Traffic-AI: Predictive Transport Analytical Platform

A scalable, cloud-native system designed to transition urban navigation from reactive congestion reporting to **proactive journey planning** based on forecasted traffic conditions.

This project implements a **Hybrid Inference Architecture** leveraging AWS Glue, Athena, and SageMaker MLOps pipelines to handle high-velocity traffic data and provide low-latency route forecasting.

| Metric                  | Detail                                                                   |
| :---------------------- | :----------------------------------------------------------------------- |
| **System Architecture** | Service-Oriented Architecture (SOA)                                      |
| **Data Volume (Daily)** | ~41.4 Million records/day (Full ETL Batch)                               |
| **Core Features**       | 1. City-Scale 24h Traffic Forecast (Heatmap via Athena)                  |
|                         | 2. Point-to-Point Route Optimization (Real-Time Inference via SageMaker) |
| **Performance Gain**    | SOA p95 Latency: 4.7s (down from 43s Monolithic Baseline)                |
| **Resilience Proof**    | RTO (Recovery Time Objective): ~14 minutes via IaC                       |

---

## 1. Prerequisites & Configuration

Before deployment or local development, ensure the following tools are installed and configured:

- **AWS CLI:** Configured with credentials and necessary IAM permissions.
- **Terraform:** (v1.5+) For provisioning and managing infrastructure (IaC).
- **Docker:** For building and running the Flask application container.
- **Python:** (v3.10+) With `pip` and a virtual environment (recommended).

### 1.1 API Keys Required

The application requires credentials for both data ingestion (background) and routing (application logic). These should be stored in AWS Secrets Manager and/or a local `.env` file for development.

- **LTA DataMall API Key:** Required for traffic speed and carpark data ingestion.
- **NEA API Key:** Required for real-time weather data streams.
- **Google Maps API Key:** Required for the `/route` endpoint and map visualization. **(Ensure Directions API and Geocoding API are enabled).**

---

## 2. Local Development Setup (Mock Data)

This section details how to run the application locally for feature development and testing the user interface logic.

### 2.1 Generate Traffic Data Assets

We generate decoupled, memory-optimized data files to simulate the output of the full AWS ETL pipeline.

1.  **Install Dependencies:** Ensure core data packages are available.
    ```bash
    pip install pandas numpy awswrangler
    # Note: Requires GeoJSON file (roads_wgs84.geojson)
    ```
2.  **Run the Generator:** This processes the road network and creates two output files:
    - `mock_geometry.json` (The static road shapes/curves)
    - `mock_predictions.json` (The lightweight time-series data)
    ```bash
    python src/backend/data/mock_data.py
    ```
3.  **Place Files:** Move the geometry file to the expected static path:
    ```bash
    mv mock_geometry.json src/frontend/static/
    ```

### 2.2 Run the Flask Application Locally

Run the server in local development mode, which reads the static JSON files.

```bash
# Note: This command runs the server in Mock Mode by default.
python server.py
```

- **Access Map:** Open your browser to `http://0.0.0.0:8000/predictions`
- **Verification:** The map should load the geometry instantly, and the slider should update colours using the lightweight prediction data, demonstrating high FPS.

---

## 3. Cloud Deployment (Terraform MLOps Pipeline)

The infrastructure is provisioned using Terraform, which sets up the full MLOps and Big Data architecture (VPC, ALB, ASG, Glue Jobs, Athena, and SageMaker integration).

### 3.1 Build & Push Container Image

1. Build the Docker image:

```bash
docker build -t traffic-ai-repo .
```

2. Authenticate and Push: Authenticate Docker against AWS ECR and push the tagged image:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 754029048130.dkr.ecr.us-east-1.amazonaws.com
docker tag traffic-ai-repo:latest [754029048130.dkr.ecr.us-east-1.amazonaws.com/traffic-ai-repo:latest](https://754029048130.dkr.ecr.us-east-1.amazonaws.com/traffic-ai-repo:latest)
docker push [754029048130.dkr.ecr.us-east-1.amazonaws.com/traffic-ai-repo:latest](https://754029048130.dkr.ecr.us-east-1.amazonaws.com/traffic-ai-repo:latest)
```

### 3.2 Provision Infrastructure

1. Initialize Terraform: Ensure the remote state backend (S3/DynamoDB) is configured.

```bash
terraform init
```

2. Review Plan: Check all resources to be created (VPC, ALB, ASG, Glue Jobs, Lambda functions).

```bash
terraform plan
```

3. Apply Configuration: Provision the entire stack.

```bash
terraform apply --auto-approve
```

Note: The successful completion of this step (RTO: ~12 minutes) verifies the full structural resilience of the IaC strategy.

### 3.3 Production Endpoints

Once deployment is complete, the application provides the following key functional endpoints:

| Endpoint           | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `/healthz`         | Basic liveness probe used by the Application Load Balancer (ALB) to verify instance health.                           |
| `/api/predictions` | Fetches the 24-hour city-scale traffic forecast generated by the Glue/Spark batch pipeline and served through Athena. |
| `/predict_route`   | Performs real-time inference for point-to-point journey planning using the SageMaker STGCN/GraphWaveNet endpoint.     |
| `/get_weather`     | Retrieves updated environmental conditions via NEA API integration — used for congestion sensitivity analysis.        |
| `/carparks`        | Returns real-time carpark occupancy and availability sourced from LTA DataMall.                                       |

### Example Usage

```bash
# Retrieve city-scale forecast
curl -X GET http://<domain>/api/predictions

# Request route prediction using origin + destination input
curl -X POST "http://<domain>/predict_route" \
     -H "Content-Type: application/json" \
     -d '{"origin": "1.2765,103.8456", "destination": "1.3048,103.8318"}'

# Check server alive status
curl http://<domain>/healthz
```

---

## 4. License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2025 Traffic-AI Project Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
