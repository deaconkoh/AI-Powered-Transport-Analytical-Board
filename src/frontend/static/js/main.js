// ================= MAP INIT =================
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
      },
    },
    layers: [
      { id: "osm", type: "raster", source: "osm", minzoom: 0, maxzoom: 19 },
    ],
  },
  center: [103.8198, 1.3521],
  zoom: 11,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");

// ================ STATE =====================
let originMarker = null;
let destMarker = null;
let carparkMarkers = [];
let showingCarparks = false;
let showingForecast = false;
let lastDestCoords = null;
let currentForecastHour = 0;

let roadsData = null; // mutable copy
let roadsDataOriginal = null; // pristine copy to reset
let activeCarparkPopup = null;
let boundCloseOnMapClick = false;

// ================= DOM HOOKS ================
const forecastBtn = document.getElementById("forecastBtn");
const carparkBtn = document.getElementById("carparkBtn");
const forecastControls = document.getElementById("forecastControls");
const forecastSlider = document.getElementById("forecastSlider");
const forecastTime = document.getElementById("forecastTime");
const forecastOffset = document.getElementById("forecastOffset");

const tollToggle = document.getElementById("tollToggle");
const tollCard = document.getElementById("tollCard");
const highwayToggle = document.getElementById("highwayToggle");
const highwayCard = document.getElementById("highwayCard");
const fastestToggle = document.getElementById("fastestToggle");
const fastestCard = document.getElementById("fastestCard");
const calculateBtn = document.getElementById("calculateBtn");
const errorMessage = document.getElementById("errorMessage");
const routeInfo = document.getElementById("routeInfo");
const carparkOverlay = document.getElementById("carparkOverlay");

// ================= LOAD BASE ROAD NETWORK =================
const neutralRoadColorExpr = [
  "match",
  ["get", "RD_CATG__1"],
  "Category 1",
  "#ff6b6b",
  "Category 2",
  "#ff8e4d",
  "Category 3",
  "#ffd166",
  "Category 4",
  "#7bd389",
  "Category 5",
  "#4db6ac",
  "Category 6",
  "#64b5f6",
  "Category 7",
  "#ba68c8",
  "Category 8",
  "#9575cd",
  /* default */ "#aaaaaa",
];

// Forecast gradient (match the card): green → yellow → orange → red
// NOTE: LTA band: 8 fast (low congestion, green) ... 1 slow (heavy, red)
const forecastColorExpr = [
  "interpolate",
  ["linear"],
  ["to-number", ["coalesce", ["get", "band"], 8]],
  1,
  "#ef4444", // red (heavy)
  3,
  "#f97316", // orange
  5,
  "#eab308", // yellow
  8,
  "#22c55e", // green (low)
];

// Thicker for worse congestion (band 1 thick → band 8 thin)
const forecastWidthExpr = [
  "interpolate",
  ["linear"],
  ["to-number", ["coalesce", ["get", "band"], 8]],
  1,
  4.8,
  8,
  1.6,
];

map.on("load", async () => {
  try {
    const res = await fetch("../static/js/roads_wgs84.geojson"); // put file next to this JS
    if (!res.ok) throw new Error("Failed to load road network GeoJSON");
    const roads = await res.json();

    // keep copies so we can mutate/reset
    roadsDataOriginal = roads;
    roadsData = JSON.parse(JSON.stringify(roads));

    map.addSource("roads", {
      type: "geojson",
      data: roadsData,
      lineMetrics: true,
    });

    // casing
    map.addLayer({
      id: "roads-case",
      type: "line",
      source: "roads",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": "#ffffff",
        "line-opacity": 0.55,
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.6, 14, 3.2],
      },
    });

    // base (neutral) view by category
    map.addLayer({
      id: "roads-line",
      type: "line",
      source: "roads",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": neutralRoadColorExpr,
        "line-opacity": 0.95,
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.2, 14, 2.4],
      },
    });

    // Fit to roads if turf available
    try {
      if (typeof turf !== "undefined") {
        const bbox = turf.bbox(roadsData);
        map.fitBounds(bbox, { padding: 40, duration: 0 });
      }
    } catch (e) {
      console.warn("turf bbox failed, skipping auto-fit:", e);
    }
  } catch (err) {
    console.error("Failed to load roads layer:", err);
  }
});

// ============ SMALL UTILITIES ===============
function parseLatLng(str) {
  const parts = (str || "")
    .trim()
    .split(",")
    .map((s) => s.trim());
  if (parts.length !== 2) return null;
  const lat = parseFloat(parts[0]),
    lng = parseFloat(parts[1]);
  return Number.isFinite(lat) &&
    Number.isFinite(lng) &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180
    ? [lat, lng]
    : null;
}
function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.add("visible");
  setTimeout(() => errorMessage.classList.remove("visible"), 5000);
}
function formatDuration(seconds) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60),
    mins = minutes % 60;
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}
function addPin(lngLat, htmlLabel) {
  const el = document.createElement("div");
  el.className = "emoji-marker";
  el.textContent = "📍";
  el.style.fontSize = "34px";
  el.style.lineHeight = "1";
  el.style.filter = "drop-shadow(0 1px 1px rgba(0,0,0,.35))";
  return new maplibregl.Marker({
    element: el,
    anchor: "bottom",
    offset: [0, 6],
  })
    .setLngLat(lngLat)
    .setPopup(new maplibregl.Popup({ offset: 12 }).setHTML(htmlLabel))
    .addTo(map);
}

// ============ TOGGLES ===========
function setupToggle(card, toggle, activeClass = "active") {
  const toggleSwitch = () => {
    toggle.classList.toggle(activeClass);
    card.classList.toggle(activeClass);
  };
  card.addEventListener("click", toggleSwitch);
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSwitch();
  });
}
setupToggle(tollCard, tollToggle, "active");
setupToggle(highwayCard, highwayToggle, "active");
setupToggle(fastestCard, fastestToggle, "active-purple");

// ========== FORECAST: JOIN SPEEDBANDS → ROADS =========
function normName(s) {
  return (s || "").toUpperCase().replace(/\s+/g, " ").trim();
}

// Apply worst band per road **by name** (fast POC)
function applyBandsToRoads_ByName(speedRows) {
  if (!roadsData) return;

  // Collect "worst" (i.e., smallest) band per road name
  const worstByName = new Map();
  for (const r of speedRows) {
    const name = normName(r.RoadName);
    const b = Math.max(1, Math.min(8, Number(r.SpeedBand) || 0));
    if (!name || !b) continue;
    const curr = worstByName.get(name);
    worstByName.set(name, curr ? Math.min(curr, b) : b);
  }

  // Write band onto features (property 'band')
  for (const f of roadsData.features) {
    const name = normName(f.properties.RD_CD_DESC);
    f.properties.band = worstByName.get(name) ?? null;
  }

  // Push back + switch to forecast paint (gradient + width)
  const src = map.getSource("roads");
  if (src) src.setData(roadsData);
  if (map.getLayer("roads-line")) {
    map.setPaintProperty("roads-line", "line-color", forecastColorExpr);
    map.setPaintProperty("roads-line", "line-width", [
      "interpolate",
      ["linear"],
      ["zoom"],
      10,
      ["*", 0.9, forecastWidthExpr], // scale at low zoom
      14,
      forecastWidthExpr,
    ]);
    map.setPaintProperty("roads-line", "line-opacity", 0.98);
  }
  // slightly bump casing for clarity
  if (map.getLayer("roads-case")) {
    map.setPaintProperty("roads-case", "line-width", [
      "interpolate",
      ["linear"],
      ["zoom"],
      10,
      2.0,
      14,
      4.0,
    ]);
  }
}

function resetRoadStyleToBase() {
  if (!roadsDataOriginal || !map.getSource("roads")) return;
  roadsData = JSON.parse(JSON.stringify(roadsDataOriginal)); // drop 'band'
  map.getSource("roads").setData(roadsData);
  if (map.getLayer("roads-line")) {
    map.setPaintProperty("roads-line", "line-color", neutralRoadColorExpr);
    map.setPaintProperty("roads-line", "line-width", [
      "interpolate",
      ["linear"],
      ["zoom"],
      10,
      1.2,
      14,
      2.4,
    ]);
    map.setPaintProperty("roads-line", "line-opacity", 0.95);
  }
  if (map.getLayer("roads-case")) {
    map.setPaintProperty("roads-case", "line-width", [
      "interpolate",
      ["linear"],
      ["zoom"],
      10,
      1.6,
      14,
      3.2,
    ]);
  }
}

// Forecast UI
forecastBtn.addEventListener("click", async () => {
  showingForecast = !showingForecast;
  forecastBtn.classList.toggle("active", showingForecast);
  forecastControls.classList.toggle("visible", showingForecast);

  // turn off carparks if forecast is on
  if (showingForecast && showingCarparks) {
    showingCarparks = false;
    carparkBtn.classList.remove("active");
    carparkOverlay.classList.remove("visible");
    carparkMarkers.forEach((m) => m.remove());
  }

  if (showingForecast) {
    try {
      const resp = await fetch(`/speedbands?hour=${currentForecastHour}`);
      if (!resp.ok) throw new Error("speedbands fetch failed");
      const rows = (await resp.json()) || [];
      applyBandsToRoads_ByName(rows);
    } catch (e) {
      console.warn("speedbands unavailable:", e);
    }
  } else {
    resetRoadStyleToBase();
  }
});

forecastSlider.addEventListener("input", async (e) => {
  currentForecastHour = parseInt(e.target.value, 10);
  const now = new Date();
  const forecastDate = new Date(
    now.getTime() + currentForecastHour * 3600 * 1000
  );
  forecastTime.textContent = forecastDate.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  forecastOffset.textContent =
    currentForecastHour === 0 ? "Now" : `+${currentForecastHour}h`;
  if (showingForecast) {
    try {
      const resp = await fetch(`/speedbands?hour=${currentForecastHour}`);
      if (resp.ok) applyBandsToRoads_ByName(await resp.json());
    } catch (e) {
      console.warn(e);
    }
  }
});
// init forecast display text
(function () {
  const now = new Date();
  forecastTime.textContent = now.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  forecastOffset.textContent = "Now";
})();

// ========== CARPARK TOGGLE ==========
carparkBtn.addEventListener("click", () => {
  showingCarparks = !showingCarparks;
  carparkBtn.classList.toggle("active", showingCarparks);
  carparkOverlay.classList.toggle("visible", showingCarparks);

  if (showingCarparks && showingForecast) {
    showingForecast = false;
    forecastBtn.classList.remove("active");
    forecastControls.classList.remove("visible");
    resetRoadStyleToBase();
  }

  if (showingCarparks) {
    // Prefer last destination; otherwise request ALL
    if (Array.isArray(lastDestCoords)) {
      const [lng, lat] = lastDestCoords;
      fetchCarparksNear(lat, lng, 1.0);
    } else {
      fetchCarparksNear(); // no args => get ALL
    }
  } else {
    carparkMarkers.forEach((m) => m.remove());
  }
});

// ========== ROUTING (unchanged from your last) =========
document.getElementById("routeForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const originInput = document.getElementById("origin").value.trim();
  const destInput = document.getElementById("dest").value.trim();
  if (!originInput || !destInput)
    return showError("Please enter both origin and destination");

  let origin = parseLatLng(originInput);
  let dest = parseLatLng(destInput);
  if (!origin) origin = originInput;
  if (!dest) dest = destInput;

  const filters = {
    avoidERP: tollToggle.classList.contains("active"),
    avoidHighway: highwayToggle.classList.contains("active"),
    fastest: fastestToggle.classList.contains("active-purple"),
  };

  const originalBtnContent = calculateBtn.innerHTML;
  calculateBtn.innerHTML = '<span class="loading"></span> Calculating...';
  calculateBtn.disabled = true;
  errorMessage.classList.remove("visible");

  try {
    const resp = await fetch("/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, destination: dest, filters }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Route calculation failed");
    if (!data.overview_polyline) throw new Error("No route polyline returned");

    const coords = polyline
      .decode(data.overview_polyline)
      .map(([lat, lng]) => [lng, lat]);

    if (map.getLayer("route-layer")) map.removeLayer("route-layer");
    if (map.getSource("route-source")) map.removeSource("route-source");
    if (originMarker) originMarker.remove();
    if (destMarker) destMarker.remove();

    map.addSource("route-source", {
      type: "geojson",
      data: {
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords },
      },
    });
    map.addLayer({
      id: "route-layer",
      type: "line",
      source: "route-source",
      paint: { "line-color": "#667eea", "line-width": 5, "line-opacity": 0.8 },
    });

    const startLL = Array.isArray(origin) ? [origin[1], origin[0]] : coords[0];
    const endLL = Array.isArray(dest)
      ? [dest[1], dest[0]]
      : coords[coords.length - 1];
    originMarker = addPin(startLL, "<b>📍 Start</b>");
    destMarker = addPin(endLL, "<b>📍 Destination</b>");

    const bounds = coords.reduce(
      (b, c) => b.extend(c),
      new maplibregl.LngLatBounds(coords[0], coords[0])
    );
    map.fitBounds(bounds, { padding: 50 });

    document.getElementById("distanceValue").textContent =
      data.distance_km.toFixed(1);
    document.getElementById("durationValue").textContent = formatDuration(
      data.eta_seconds
    );
    routeInfo.classList.add("visible");

    const endPt = coords[coords.length - 1];
    lastDestCoords = endPt;
    fetchWeather({ lat: endPt[1], lon: endPt[0] });

    // Show the "Open in Google Maps" button
    const googleMapsBtn = document.getElementById("googleMapsBtn");
    if (googleMapsBtn) {
      googleMapsBtn.style.display = "inline-flex"; // show the button
      googleMapsBtn.onclick = () => {
        const start = coords[0];
        const dest = coords[coords.length - 1];

        // Detect iOS or Android and open appropriate map app
        let url;
        if (/iPhone|iPad|Macintosh/i.test(navigator.userAgent)) {
          url = `http://maps.apple.com/?saddr=${start[1]},${start[0]}&daddr=${dest[1]},${dest[0]}&dirflg=d`;
        } else {
          url = `https://www.google.com/maps/dir/?api=1&origin=${start[1]},${start[0]}&destination=${dest[1]},${dest[0]}&travelmode=driving`;
        }
        window.open(url, "_blank");
      };
    }

    // If carpark overlay is active, refresh markers near destination immediately
    if (showingCarparks && Array.isArray(lastDestCoords)) {
      const [lng, lat] = lastDestCoords;
      fetchCarparksNear(lat, lng, 1.0);
    }

    const startCoords = Array.isArray(origin)
      ? origin
      : [coords[0][1], coords[0][0]];
    const destinationCoords = Array.isArray(dest)
      ? dest
      : [coords.at(-1)[1], coords.at(-1)[0]];
    const aiPrediction = await fetchAIPrediction(
      startCoords,
      destinationCoords
    );
    displayAIPredictions(aiPrediction);
  } catch (err) {
    console.error(err);
    showError(err.message || "Network error occurred");
  } finally {
    calculateBtn.innerHTML = originalBtnContent;
    calculateBtn.disabled = false;
  }
});

// ========== WEATHER ===========
async function fetchWeather({ city, lat, lon } = {}) {
  try {
    const qs = city
      ? `?city=${encodeURIComponent(city)}`
      : Number.isFinite(lat) && Number.isFinite(lon)
      ? `?lat=${lat}&lon=${lon}`
      : "";
    const response = await fetch(`/get_weather${qs}`);
    const data = await response.json();
    const weatherWidget = document.getElementById("weatherWidget");

    const getWeatherIcon = (condition) => {
      const cond = (condition || "").toLowerCase();
      if (cond.includes("rain") || cond.includes("showers")) return "🌧️";
      if (cond.includes("thunder")) return "⛈️";
      if (cond.includes("cloud")) return "☁️";
      if (cond.includes("sun") || cond.includes("fair")) return "☀️";
      if (cond.includes("hazy") || cond.includes("haze")) return "🌫️";
      return "🌤️";
    };

    const locationLabel =
      data.resolved_area ||
      city ||
      (Number.isFinite(lat) && Number.isFinite(lon)
        ? `${lat.toFixed(4)}, ${lon.toFixed(4)}`
        : "Singapore");

    weatherWidget.innerHTML = `
      <div class="weather-header">
        <div class="weather-main">
          <div class="weather-icon">${getWeatherIcon(
            data.forecast || "Fair"
          )}</div>
          <div><div class="weather-temp">${Math.round(
            Number.isFinite(+data.temperature) ? +data.temperature : 30
          )}<span class="weather-unit">°C</span></div></div>
        </div>
        <div style="text-align:right;">
          <div class="weather-location">📍 ${locationLabel}</div>
          <div class="weather-condition">${
            data.forecast || "Fair Weather"
          }</div>
        </div>
      </div>
      <div class="weather-details">
        <div class="weather-detail-item"><div class="weather-detail-icon">💧</div><div class="weather-detail-value">${
          Number.isFinite(+data.humidity) ? data.humidity : 75
        }%</div><div class="weather-detail-label">Humidity</div></div>
        <div class="weather-detail-item"><div class="weather-detail-icon">💨</div><div class="weather-detail-value">${
          Number.isFinite(+data.wind_speed) ? data.wind_speed : 15
        } km/h</div><div class="weather-detail-label">Wind</div></div>
        <div class="weather-detail-item"><div class="weather-detail-icon">🌡️</div><div class="weather-detail-value">${
          Number.isFinite(+data.feels_like) ? data.feels_like : 32
        }°C</div><div class="weather-detail-label">Feels Like</div></div>
      </div>`;
  } catch (error) {
    console.error("Error fetching weather:", error);
    document.getElementById(
      "weatherWidget"
    ).innerHTML = `<div class="weather-loading"><div>❌ Unable to load weather</div></div>`;
  }
}
fetchWeather();
setInterval(fetchWeather, 600000);

// ========== CARPARKS ==========
async function fetchCarparksNear(lat, lon, radiusKm = 1.0) {
  const carparkList = document.getElementById("carparkList");
  const countEl = document.getElementById("carparkCount");

  try {
    // clear existing
    carparkMarkers.forEach((m) => m.remove());
    carparkMarkers = [];
    if (activeCarparkPopup) {
      activeCarparkPopup.remove();
      activeCarparkPopup = null;
    }

    carparkList.innerHTML = `<div class="carpark-loading">🔎 Finding carparks near your destination…</div>`;

    // build query
    const hasCoords = Number.isFinite(lat) && Number.isFinite(lon);
    const qs = hasCoords ? `?lat=${lat}&lon=${lon}&radius_km=${radiusKm}` : "";
    const resp = await fetch(`/carparks${qs}`);
    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.error || "Failed to load carparks");

    const carparks = payload.carparks || [];
    countEl.textContent = hasCoords
      ? `${carparks.length} carpark${
          carparks.length !== 1 ? "s" : ""
        } within ${radiusKm} km`
      : `${carparks.length} carpark${carparks.length !== 1 ? "s" : ""}`;
    carparkList.innerHTML = carparks.length
      ? ""
      : `<div class="carpark-loading">No carparks found${
          hasCoords ? " nearby." : "."
        }</div>`;

    // render markers + list
    carparks.forEach((cp) => {
      const { Latitude, Longitude } = cp.Location;

      // Determine availability class
      let badgeClass = "available";
      let badgeText = `${cp.AvailableLots ?? "—"} lots`;
      const lots = Number(cp.AvailableLots);
      if (Number.isFinite(lots)) {
        if (lots < 10) {
          badgeClass = "full";
          badgeText = "Almost Full";
        } else if (lots < 30) {
          badgeClass = "limited";
        }
      }

      // 🅿️ marker element (uses .custom-marker CSS)
      const el = document.createElement("div");
      el.className = "custom-marker";
      el.textContent = "🅿️";

      // Popup (with ❌ button)
      const popup = new maplibregl.Popup({
        offset: 18,
        closeButton: true, // ✅ show X button
        closeOnClick: false, // we manage this manually
      }).setHTML(`
        <b>${cp.Development || "Carpark"}</b><br>
        Available: ${cp.AvailableLots ?? "—"}<br>
        Type: ${cp.LotType === "C" ? "Car" : cp.LotType || "—"}<br>
        Agency: ${cp.Agency || "—"}<br>
        ${cp.DistanceKm != null ? `Distance: ${cp.DistanceKm} km` : ""}
      `);

      const marker = new maplibregl.Marker({ element: el, anchor: "bottom" })
        .setLngLat([Longitude, Latitude])
        .setPopup(popup)
        .addTo(map);
      carparkMarkers.push(marker);

      // Marker click → open this popup, close others
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        if (activeCarparkPopup && activeCarparkPopup !== popup) {
          activeCarparkPopup.remove();
        }
        popup.addTo(map);
        activeCarparkPopup = popup;
      });

      // When user manually closes with ❌
      popup.on("close", () => {
        if (activeCarparkPopup === popup) {
          activeCarparkPopup = null;
        }
      });

      // Sidebar list item
      const item = document.createElement("div");
      item.className = "carpark-item";
      item.innerHTML = `
        <div class="carpark-item-header">
          <div class="carpark-name">${cp.Development || "Carpark"}</div>
          <div class="carpark-availability">
            <div class="availability-badge ${badgeClass}">${badgeText}</div>
          </div>
        </div>
        <div class="carpark-info">
          <div class="carpark-info-item"><span>📍</span><span>${
            cp.Area || "Nearby"
          }</span></div>
          <div class="carpark-info-item"><span>🚗</span><span>${
            cp.LotType === "C" ? "Car" : cp.LotType || "—"
          }</span></div>
          <div class="carpark-info-item"><span>🏢</span><span>${
            cp.Agency || "—"
          }</span></div>
          ${
            cp.DistanceKm != null
              ? `<div class="carpark-info-item"><span>📏</span><span>${cp.DistanceKm} km</span></div>`
              : ""
          }
        </div>
      `;

      item.addEventListener("click", () => {
        map.flyTo({ center: [Longitude, Latitude], zoom: 16 });
        if (activeCarparkPopup && activeCarparkPopup !== popup) {
          activeCarparkPopup.remove();
        }
        popup.addTo(map);
        activeCarparkPopup = popup;
      });

      carparkList.appendChild(item);
    });
  } catch (error) {
    console.error("Error fetching carparks:", error);
    carparkList.innerHTML = `<div class="carpark-loading">❌ Unable to load carpark data. ${
      error.message || ""
    }</div>`;
  }
}

// ========== AI PREDICTION ==========
async function fetchAIPrediction(originCoords, destCoords) {
  try {
    const response = await fetch("/predict_route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin: originCoords, destination: destCoords }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.warn(
        "AI prediction not available:",
        errorData.error || response.status
      );
      return null;
    }
    return await response.json();
  } catch (error) {
    console.warn("AI prediction failed:", error);
    return null;
  }
}
function displayAIPredictions(prediction) {
  if (!prediction) return;
  let aiSection = document.getElementById("aiPredictions");
  if (!aiSection) {
    aiSection = document.createElement("div");
    aiSection.id = "aiPredictions";
    aiSection.className = "route-info";
    aiSection.innerHTML = `<div class="route-info-title">🤖 AI Traffic Predictions</div><div id="aiPredictionsContent"></div>`;
    routeInfo.parentNode.insertBefore(aiSection, routeInfo.nextSibling);
  }

  const content = document.getElementById("aiPredictionsContent");
  if (prediction.error) {
    content.innerHTML = `<div style="color:#e53e3e;font-size:0.875rem;">⚠️ AI prediction temporarily unavailable</div>`;
    return;
  }

  const trafficConditions = prediction.traffic_conditions || {};
  const aiInsights = prediction.ai_insights || {};
  content.innerHTML = `
    <div class="route-stats">
      <div class="stat-item">
        <div class="stat-value" style="color:${
          trafficConditions.predicted_congestion_level === "HEAVY"
            ? "#e53e3e"
            : trafficConditions.predicted_congestion_level === "MODERATE"
            ? "#d69e2e"
            : "#38a169"
        }">${trafficConditions.predicted_congestion_level || "MODERATE"}</div>
        <div class="stat-label">Congestion Level</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">${Math.round(
          (trafficConditions.overall_confidence ?? 0.5) * 100
        )}%</div>
        <div class="stat-label">Confidence</div>
      </div>
    </div>
    ${
      aiInsights.risk_factors && aiInsights.risk_factors.length
        ? `<div style="margin-top:1rem;padding:.75rem;background:#fffaf0;border-radius:8px;border-left:4px solid #ed8936;">
             <div style="font-size:.8125rem;font-weight:600;color:#744210;margin-bottom:.5rem;">⚠️ Risk Factors</div>
             <ul style="font-size:.75rem;color:#744210;margin:0;padding-left:1rem;">
               ${aiInsights.risk_factors.map((f) => `<li>${f}</li>`).join("")}
             </ul>
           </div>`
        : ""
    }
    ${
      trafficConditions.recommended_departure_time
        ? `<div style="margin-top:.75rem;font-size:.8125rem;color:#4a5568;">
             <span style="font-weight:600;">🕒 Recommended departure:</span> ${trafficConditions.recommended_departure_time}
           </div>`
        : ""
    }
    ${
      prediction.models_used && prediction.models_used.length
        ? `<div style="margin-top:.5rem;font-size:.75rem;color:#718096;">AI Models used: ${prediction.models_used.join(
            ", "
          )}</div>`
        : ""
    }`;
  aiSection.classList.add("visible");
}
