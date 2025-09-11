"""
LTA DataMall API Client
=======================

Usage:
from lta_client import client, LTAClient

Option A: Quick start (singleton)
c = client()
print(c.traffic_incidents())

Option B: Create your own instance
c = LTAClient()
print(c.bus_arrivals("01019"))   # Fullerton Hotel bus stop

Available methods:
Bus:
  - bus_arrivals(bus_stop_code: str)
  - bus_services(skip: int = 0)
  - bus_stops(skip: int = 0)

Train:
  - train_service_alerts()
  - station_crowd_density_realtime(train_line: str)
  - station_crowd_density_forecast(train_line: str)

Road / Car:
  - traffic_incidents(skip: int = 0)
  - traffic_speed_bands(skip: int = 0)
  - traffic_images()
  - est_travel_times(skip: int = 0)
  - approved_road_works()

Notes:
- API key must be set in `.env` as LTA_ACCOUNT_KEY. (Follow .env.example)
- Most responses contain a "value" list of results.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional, Callable, List
import requests

from dotenv import load_dotenv
load_dotenv()


LTA_BASE_DEFAULT = "https://datamall2.mytransport.sg/ltaodataservice"
DEFAULT_TIMEOUT = (3.05, 15) # (connect timeout, read timeout)

class LTAError(RuntimeError): # Raised when an LTA API call fails
    pass

class LTAClient:
    """
    Thin wrapper around LTA DataMall.
    """
    def __init__(
        self,
        account_key: Optional[str] = None,
        base_url: str = LTA_BASE_DEFAULT,
        session: Optional[requests.Session] = None,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    ) -> None:
        self.account_key = account_key or os.getenv("LTA_ACCOUNT_KEY")
        if not self.account_key:
            raise LTAError("Missing LTA account key. Set LTA_ACCOUNT_KEY env or pass account_key.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
    
    # ---------- internal methods ----------

    def _headers(self) -> Dict[str, str]:
        return {
            "AccountKey": self.account_key,
            "accept": "application/json",
        }

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            # Keep a concise error with context (status + endpoint)
            status = getattr(e.response, "status_code", "unknown")
            msg = getattr(e.response, "text", "")
            raise LTAError(f"HTTP {status} for {url} params={params} :: {msg[:200]}") from e
        except requests.RequestException as e:
            raise LTAError(f"Network error calling {url}: {e}") from e

    # ---------- bus ----------

    def bus_arrivals(self, bus_stop_code: str) -> Dict[str, Any]:
        """Real-time bus arrivals at a stop. Endpoint: /BusArrival"""
        return self._request("v3/BusArrival", params={"BusStopCode": bus_stop_code})

    def bus_services(self, skip: int = 0) -> Dict[str, Any]:
        """All bus services (routes, operators). Endpoint: /BusServices"""
        params = {"$skip": skip} if skip else None
        return self._request("BusServices", params=params)

    def bus_stops(self, skip: int = 0) -> Dict[str, Any]:
        """Bus stops metadata (codes, names, coords). Endpoint: /BusStops"""
        params = {"$skip": skip} if skip else None
        return self._request("BusStops", params=params)

    # ---------- train ----------

    def train_service_alerts(self) -> Dict[str, Any]:
        """MRT service alerts/disruptions. Endpoint: /TrainServiceAlerts"""
        return self._request("TrainServiceAlerts")
    
    def station_crowd_density_realtime(self, train_line: str) -> Dict[str, Any]:
        """
        Real-time station crowd levels for a train line.
        TrainLine codes: CCL, CEL, CGL, DTL, EWL, NEL, NSL, BPL, SLRT, PLRT, TEL
        Endpoint: /PCDRealTime
        """
        return self._request("PCDRealTime", params={"TrainLine": train_line})
    
    def station_crowd_density_forecast(self, train_line: str) -> Dict[str, Any]:
        """
        Forecast station crowd levels (30-min intervals) for a train line.
        Endpoint: /PCDForecast
        """
        return self._request("PCDForecast", params={"TrainLine": train_line})



    # ---------- road / car ----------

    def traffic_incidents(self, skip: int = 0) -> Dict[str, Any]:
        """Incidents: accidents, breakdowns, roadworks. Endpoint: /TrafficIncidents"""
        params = {"$skip": skip} if skip else None
        return self._request("TrafficIncidents", params=params)

    def traffic_speed_bands(self, skip: int = 0) -> Dict[str, Any]:
        """Segment speeds (congestion). Endpoint: /TrafficSpeedBands"""
        params = {"$skip": skip} if skip else None
        return self._request("v4/TrafficSpeedBands", params=params)

    def traffic_images(self) -> Dict[str, Any]:
        """Live traffic camera snapshots. Endpoint: /Traffic-Imagesv2"""
        return self._request("Traffic-Imagesv2")

    def est_travel_times(self, skip: int = 0) -> Dict[str, Any]:
        """Estimated expressway travel times (minutes). Endpoint: /EstTravelTimes"""
        params = {"$skip": skip} if skip else None
        return self._request("EstTravelTimes", params=params)
    
    def approved_road_works(self) -> Dict[str, Any]:
        """Approved road works (planned/ongoing). Endpoint: /RoadWorks"""
        return self._request("RoadWorks")


    # ---------- helpers ----------

    def paged(self, fn: Callable[..., Dict[str, Any]], page_size: int = 500, max_pages: int = 20) -> List[Dict[str, Any]]:
        """
        Generic $skip paging helper. Call with a method like `client.bus_stops`.
        Returns a concatenated list from 'value'.
        """
        out: List[Dict[str, Any]] = []
        skip = 0
        for _ in range(max_pages):
            data = fn(skip=skip)
            chunk = data.get("value", [])
            if not chunk:
                break
            out.extend(chunk)
            if len(chunk) < page_size:
                break
            skip += page_size
        return out

_client: Optional[LTAClient] = None

def client() -> LTAClient:
    """Singleton accessor for quick usage in small scripts."""
    global _client
    if _client is None:
        _client = LTAClient()
    return _client

