// Init map
const map = L.map("map").setView([1.3521, 103.8198], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
}).addTo(map);

// Draw controls (polygon)
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
map.addControl(
  new L.Control.Draw({
    draw: {
      polygon: true,
      polyline: false,
      rectangle: false,
      circle: false,
      marker: false,
      circlemarker: false,
    },
    edit: { featureGroup: drawnItems },
  })
);
map.on(L.Draw.Event.CREATED, (e) => drawnItems.addLayer(e.layer));

let currentRouteLayer = null;
let originMarker = null;
let destMarker = null;

// DOM elements
const tollToggle = document.getElementById("tollToggle");
const tollCard = document.getElementById("tollCard");
const highwayToggle = document.getElementById("highwayToggle");
const highwayCard = document.getElementById("highwayCard");
const fastestToggle = document.getElementById("fastestToggle");
const fastestCard = document.getElementById("fastestCard");
const calculateBtn = document.getElementById("calculateBtn");
const errorMessage = document.getElementById("errorMessage");
const routeInfo = document.getElementById("routeInfo");
const emptyState = document.getElementById("emptyState");

// Toggle switches with card click support
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

// Helper: parse "lat,lng"
function parseLatLng(str) {
  const trimmed = (str || "").trim();
  const parts = trimmed.split(",").map((s) => s.trim());

  if (parts.length !== 2) return null;

  const lat = parseFloat(parts[0]);
  const lng = parseFloat(parts[1]);

  if (
    Number.isFinite(lat) &&
    Number.isFinite(lng) &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180
  ) {
    return [lat, lng];
  }
  return null;
}

// Show error message
function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.add("visible");
  setTimeout(() => {
    errorMessage.classList.remove("visible");
  }, 5000);
}

// Format duration
function formatDuration(seconds) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return minutes;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}

// Handle form submission
document.getElementById("routeForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const originInput = document.getElementById("origin").value.trim();
  const destInput = document.getElementById("dest").value.trim();

  if (!originInput || !destInput) {
    showError("Please enter both origin and destination");
    return;
  }

  // Try to parse as coordinates, otherwise treat as address
  let origin = parseLatLng(originInput);
  let dest = parseLatLng(destInput);

  // If not coordinates, keep as address string
  if (!origin) origin = originInput;
  if (!dest) dest = destInput;

  const filters = {
    avoidERP: tollToggle.classList.contains("active"),
    avoidHighway: highwayToggle.classList.contains("active"),
    fastest: fastestToggle.classList.contains("active-purple"),
  };

  // Update button state
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

    if (!resp.ok) {
      throw new Error(data.error || "Route calculation failed");
    }

    console.log("Route response:", data);

    if (!data.overview_polyline) {
      throw new Error("No route polyline returned");
    }

    // Hide empty state
    emptyState.classList.add("hidden");

    // Decode Google encoded polyline
    const coords = polyline
      .decode(data.overview_polyline)
      .map(([lat, lng]) => [lat, lng]);

    // Remove previous route and markers
    if (currentRouteLayer) map.removeLayer(currentRouteLayer);
    if (originMarker) map.removeLayer(originMarker);
    if (destMarker) map.removeLayer(destMarker);

    // Add route polyline
    currentRouteLayer = L.polyline(coords, {
      color: "#667eea",
      weight: 5,
      opacity: 0.8,
      smoothFactor: 1,
    }).addTo(map);

    // Add markers if coordinates available
    if (Array.isArray(origin)) {
      originMarker = L.marker(origin).addTo(map).bindPopup("<b>📍 Start</b>");
    }
    if (Array.isArray(dest)) {
      destMarker = L.marker(dest).addTo(map).bindPopup("<b>🎯 Destination</b>");
    }

    // Fit bounds to show entire route
    map.fitBounds(currentRouteLayer.getBounds(), { padding: [50, 50] });

    document.getElementById("distanceValue").textContent =
      data.distance_km.toFixed(1);
    document.getElementById("durationValue").textContent = formatDuration(
      data.eta_seconds
    );
    routeInfo.classList.add("visible");

    const destRaw = document.getElementById("dest").value.trim();
    const destCoords = parseLatLng(destRaw);

    const endPt = coords[coords.length - 1]; // [lat, lng]
    let lastDestCoords = endPt; // store it

    // Weather: prefer coords (more accurate than city name)
    if (Array.isArray(endPt)) {
      fetchWeather({ lat: endPt[0], lon: endPt[1] });
    } else {
      const destRaw = document.getElementById("dest").value.trim();
      fetchWeather({ city: destRaw });
    }

    // Carparks: always call with destination coords from the route
    carparkToggleBtn.classList.add("active");
    carparkOverlay.classList.add("visible");
    fetchCarparksNear(lastDestCoords[0], lastDestCoords[1], 1.0);
  } catch (err) {
    console.error(err);
    showError(err.message || "Network error occurred");
    emptyState.classList.remove("hidden");
  } finally {
    calculateBtn.innerHTML = originalBtnContent;
    calculateBtn.disabled = false;
  }
});

// ========== WEATHER FUNCTIONALITY ==========
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
          <div>
            <div class="weather-temp">
              ${Math.round(
                Number.isFinite(+data.temperature) ? +data.temperature : 30
              )}
              <span class="weather-unit">°C</span>
            </div>
          </div>
        </div>
        <div style="text-align: right;">
          <div class="weather-location">📍 ${locationLabel}</div>
          <div class="weather-condition">${
            data.forecast || "Fair Weather"
          }</div>
        </div>
      </div>
      <div class="weather-details">
        <div class="weather-detail-item">
          <div class="weather-detail-icon">💧</div>
          <div class="weather-detail-value">${
            Number.isFinite(+data.humidity) ? data.humidity : 75
          }%</div>
          <div class="weather-detail-label">Humidity</div>
        </div>
        <div class="weather-detail-item">
          <div class="weather-detail-icon">💨</div>
          <div class="weather-detail-value">${
            Number.isFinite(+data.wind_speed) ? data.wind_speed : 15
          } km/h</div>
          <div class="weather-detail-label">Wind</div>
        </div>
        <div class="weather-detail-item">
          <div class="weather-detail-icon">🌡️</div>
          <div class="weather-detail-value">${
            Number.isFinite(+data.feels_like) ? data.feels_like : 32
          }°C</div>
          <div class="weather-detail-label">Feels Like</div>
        </div>
      </div>
    `;
  } catch (error) {
    console.error("Error fetching weather:", error);
    document.getElementById("weatherWidget").innerHTML = `
      <div class="weather-loading">
        <div>❌ Unable to load weather</div>
      </div>
    `;
  }
}

// Fetch weather on load
fetchWeather();
// Refresh weather every 10 minutes
setInterval(fetchWeather, 600000);

let carparkMarkers = [];
let showingCarparks = false;
let lastDestCoords = null;

// ========== CARPARK FUNCTIONALITY ==========
const carparkToggleBtn = document.getElementById("carparkToggleBtn");
const carparkOverlay = document.getElementById("carparkOverlay");

async function fetchCarparksNear(lat, lon, radiusKm = 1.0) {
  const carparkList = document.getElementById("carparkList");
  const countEl = document.getElementById("carparkCount");

  try {
    // Clear existing markers
    carparkMarkers.forEach((m) => map.removeLayer(m));
    carparkMarkers = [];

    carparkList.innerHTML = `<div class="carpark-loading">🔎 Finding carparks near your destination…</div>`;

    const resp = await fetch(
      `/carparks?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`
    );
    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.error || "Failed to load carparks");

    const carparks = payload.carparks || [];
    countEl.textContent = `${carparks.length} carpark${
      carparks.length !== 1 ? "s" : ""
    } within ${radiusKm} km`;
    carparkList.innerHTML = carparks.length
      ? ""
      : `<div class="carpark-loading">No carparks found nearby.</div>`;

    carparks.forEach((cp) => {
      const { Latitude, Longitude } = cp.Location;

      const marker = L.marker([Latitude, Longitude], {
        icon: L.divIcon({
          html: "🅿️",
          className: "emoji-icon",
          iconSize: [30, 30],
        }),
      }).addTo(map);

      marker.bindPopup(`
        <b>${cp.Development || "Carpark"}</b><br>
        Available: ${cp.AvailableLots ?? "—"}<br>
        Type: ${cp.LotType === "C" ? "Car" : cp.LotType || "—"}<br>
        Agency: ${cp.Agency || "—"}<br>
        Distance: ${cp.DistanceKm} km
      `);

      carparkMarkers.push(marker);

      // Badge
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

      // List item
      const item = document.createElement("div");
      item.className = "carpark-item";
      item.innerHTML = `
        <div class="carpark-item-header">
          <div class="carpark-name">${cp.Development || "Carpark"}</div>
          <div class="carpark-availability"><div class="availability-badge ${badgeClass}">${badgeText}</div></div>
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
          <div class="carpark-info-item"><span>📏</span><span>${
            cp.DistanceKm
          } km</span></div>
        </div>
      `;
      item.addEventListener("click", () => {
        map.setView([Latitude, Longitude], 16);
        marker.openPopup();
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

carparkToggleBtn.addEventListener("click", () => {
  showingCarparks = !showingCarparks;
  carparkToggleBtn.classList.toggle("active");
  carparkOverlay.classList.toggle("visible");

  if (showingCarparks) {
    carparkToggleBtn.innerHTML = "<span>🅿️</span><span>Hide Carparks</span>";
    if (carparkMarkers.length === 0 && Array.isArray(lastDestCoords)) {
      fetchCarparksNear(lastDestCoords[0], lastDestCoords[1], 1.0);
    } else {
      carparkMarkers.forEach((m) => m.addTo(map));
    }
  } else {
    carparkToggleBtn.innerHTML = "<span>🅿️</span><span>Show Carparks</span>";
    carparkMarkers.forEach((m) => map.removeLayer(m));
  }
});
