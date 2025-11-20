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
let lastDestCoords = null;
let activeCarparkPopup = null;

// ================= DOM HOOKS ================
const carparkBtn = document.getElementById("carparkBtn");
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

// ========== CARPARK TOGGLE ==========
carparkBtn.addEventListener("click", () => {
  showingCarparks = !showingCarparks;
  carparkBtn.classList.toggle("active", showingCarparks);
  carparkOverlay.classList.toggle("visible", showingCarparks);

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

// ========== ROUTING ==========
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

    let coords;

    if (data.mode === "mock") {
      // Mock mode: backend returns { overview_polyline: { points: [[lat, lng], ...] } }
      const pts = data.overview_polyline?.points;
      if (!Array.isArray(pts) || pts.length === 0) {
        throw new Error("Mock route has no points");
      }

      // Convert [lat, lng] -> [lng, lat] for MapLibre
      coords = pts.map(([lat, lng]) => [lng, lat]);
    } else {
      // Live mode – overview_polyline might be a string or { points: "encoded" }
      let encoded = null;

      if (typeof data.overview_polyline === "string") {
        encoded = data.overview_polyline;
      } else if (
        data.overview_polyline &&
        typeof data.overview_polyline.points === "string"
      ) {
        encoded = data.overview_polyline.points;
      }

      if (!encoded) {
        throw new Error("No route polyline returned");
      }

      coords = polyline.decode(encoded).map(([lat, lng]) => [lng, lat]);
    }

    // Safety check (optional but nice)
    if (
      !Array.isArray(coords) ||
      coords.length === 0 ||
      !coords.every(
        (c) =>
          Array.isArray(c) &&
          c.length === 2 &&
          Number.isFinite(c[0]) &&
          Number.isFinite(c[1])
      )
    ) {
      throw new Error("Invalid coordinates returned from route API");
    }

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

    const startLL = coords[0]; // [lng, lat]
    const endLL = coords[coords.length - 1]; // [lng, lat]

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

    const googleMapsBtn = document.getElementById("googleMapsBtn");
    if (googleMapsBtn) {
      googleMapsBtn.style.display = "inline-flex";
      googleMapsBtn.onclick = () => {
        const start = coords[0];
        const dest = coords[coords.length - 1];

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

      const el = document.createElement("div");
      el.className = "custom-marker";
      el.textContent = "🅿️";

      const popup = new maplibregl.Popup({
        offset: 18,
        closeButton: true,
        closeOnClick: false,
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

      el.addEventListener("click", (e) => {
        e.stopPropagation();
        if (activeCarparkPopup && activeCarparkPopup !== popup) {
          activeCarparkPopup.remove();
        }
        popup.addTo(map);
        activeCarparkPopup = popup;
      });

      popup.on("close", () => {
        if (activeCarparkPopup === popup) {
          activeCarparkPopup = null;
        }
      });

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
    aiSection.innerHTML =
      '<div class="route-info-title">🤖 AI Traffic Predictions</div><div id="aiPredictionsContent"></div>';
    routeInfo.parentNode.insertBefore(aiSection, routeInfo.nextSibling);
  }

  const content = document.getElementById("aiPredictionsContent");
  if (prediction.error) {
    content.innerHTML =
      '<div style="color:#e53e3e;font-size:0.875rem;">⚠️ AI prediction temporarily unavailable</div>';
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

// ========== PREDICTIONS PAGE BUTTON ==========
document.getElementById("predictionsBtn").addEventListener("click", () => {
  window.location.href = "/predictions";
});
