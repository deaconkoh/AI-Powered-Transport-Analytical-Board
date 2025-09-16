import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key if it exists
api_key = os.getenv('DATA_GOV_SG_API_KEY')

# Set up headers
headers = {}
if api_key:
    headers['x-api-key'] = api_key
    print("Using API key")
else:
    print("No API key found, trying without authentication")

# Option 3: Try different endpoints
endpoints = [
    "https://api.data.gov.sg/v1/environment/weather-forecast",
    "https://api.data.gov.sg/v1/environment/weather-station-readings",
    "https://api.data.gov.sg/v1/environment/air-temperature",
    "https://api.data.gov.sg/v1/environment/rainfall",
    "https://api-production.data.gov.sg/v2/public/api/collections"
]

print("Testing different endpoints...\n")

for i, url in enumerate(endpoints, 1):
    try:
        print(f"{i}. Testing: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! Status: {response.status_code}")
            data = response.json()
            
            # Print first bit of data to see what we got
            if isinstance(data, dict):
                print(f"   📋 Data keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"   📋 List with {len(data)} items")
            
            # Optionally print the full response for the first working endpoint
            print(f"   📄 Sample data: {str(data)[:200]}...")
            print()
            
        else:
            print(f"   ❌ FAILED! Status: {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            print()
            
    except requests.exceptions.RequestException as e:
        print(f"   🚫 REQUEST FAILED: {e}")
        print()

print("Endpoint testing complete!")