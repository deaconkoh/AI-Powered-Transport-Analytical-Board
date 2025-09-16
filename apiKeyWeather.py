import requests
from tabulate import tabulate
        
collection_id = 1459
headers = {"x-api-key": "v2:89708f566dfbde01033214fda89d0b69a4afae9dc29f6a2a85b81166b91cc898:_1fQS5USdhujC13bJ7_AsGpjshBIWatr"}
response = requests.get("https://api.data.gov.sg/v1/environment/weather-forecast")
data = response.json()
print(data)