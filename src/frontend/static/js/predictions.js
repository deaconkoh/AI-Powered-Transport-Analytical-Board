class TrafficPredictions {
  constructor() {
    this.map = null;
    this.timelineData = {};
    this.timeKeys = [];
    this.currentIndex = 0;
    this.init();
  }

  async init() {
    await this.loadMap();
    await this.loadData();
    this.setupEventListeners();
    this.startClock();
  }

  async loadMap() {
    this.map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        sources: {
          "osm-tiles": {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [
          {
            id: "osm-tiles",
            type: "raster",
            source: "osm-tiles",
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [103.8198, 1.3521],
      zoom: 11,
    });

    await new Promise((resolve) => this.map.on("load", resolve));
    this.map.setPaintProperty("osm-tiles", "raster-opacity", 0.8);
  }

  async loadData() {
    const loadingOverlay = document.getElementById("loadingOverlay");
    try {
      if (loadingOverlay) loadingOverlay.style.display = "flex";

      // 1. Load Geometry (The Shapes) - Static File
      console.log("Loading Geometry...");
      const geoRes = await fetch("/static/mock_geometry.json");
      const geoData = await geoRes.json();

      // 2. Load Predictions (The Data) - API
      console.log("Loading Data...");
      const dataRes = await fetch("/api/predictions");
      const dataWrapper = await dataRes.json();
      const rawData = dataWrapper.predictions;

      console.log(`✅ Loaded Geometry: ${geoData.features.length} roads`);
      console.log(`✅ Loaded Data: ${rawData.length} rows`);

      // 3. Add Geometry Source ONCE
      this.map.addSource("traffic-source", {
        type: "geojson",
        data: geoData,
        promoteId: "id", // CRITICAL: Links the Data ID to the Map ID
      });

      this.map.addLayer({
        id: "traffic-layer",
        type: "line",
        source: "traffic-source",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-width": 5,
          "line-opacity": 0.8,
          // DYNAMIC COLOR LOGIC:
          // If 'traffic_color' state is set, use it. Else grey.
          "line-color": [
            "case",
            ["!=", ["feature-state", "traffic_color"], null],
            ["feature-state", "traffic_color"],
            "#cccccc", // Default Grey
          ],
        },
      });

      // 4. Process Data
      this.timelineData = this.groupBy(rawData, "t");
      this.timeKeys = Object.keys(this.timelineData).sort();

      // 5. Setup Slider
      const slider = document.getElementById("predictionSlider");
      if (slider && this.timeKeys.length > 0) {
        slider.max = 24; // Limit view to 24h
        this.updateMapState(0); // Render first frame
      }

      this.addInteractivity();
    } catch (error) {
      console.error(error);
      this.showErrorMessage(error);
    } finally {
      if (loadingOverlay) loadingOverlay.style.display = "none";
    }
  }

  groupBy(array, key) {
    return array.reduce((result, item) => {
      (result[item[key]] = result[item[key]] || []).push(item);
      return result;
    }, {});
  }

  // --- OPTIMIZED RENDER FUNCTION ---
  updateMapState(index) {
    if (!this.timeKeys || this.timeKeys.length === 0) return;
    const timeKey = this.timeKeys[index];
    const currentHourData = this.timelineData[timeKey];

    if (!currentHourData) return;

    // Batch update feature states
    currentHourData.forEach((row) => {
      // row.id = Road ID, row.c = Color, row.s = Speed, row.t = Time
      this.map.setFeatureState(
        { source: "traffic-source", id: row.id },
        {
          traffic_color: row.c,
          current_speed: row.s,
          current_time: row.t,
        }
      );
    });

    this.updateStats(currentHourData, timeKey);
  }

  updateStats(data, timeKey) {
    let avgSpeed = 0;
    if (data && data.length > 0) {
      // Use 's' because new data file uses short keys
      const total = data.reduce((sum, r) => sum + (parseFloat(r.s) || 0), 0);
      avgSpeed = total / data.length;
    }

    const displayDate = new Date();
    displayDate.setMinutes(0, 0, 0);
    displayDate.setHours(displayDate.getHours() + this.currentIndex);

    const dayPart = displayDate.toLocaleDateString("en-US", {
      weekday: "short",
    });
    const timePart = displayDate.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
    });
    const hoursOffset = this.currentIndex;
    let finalLabel =
      hoursOffset === 0
        ? `Now (${timePart})`
        : `+${hoursOffset}h (${dayPart}, ${timePart})`;

    document.getElementById("segmentCount").textContent =
      data.length.toLocaleString();
    document.getElementById("avgSpeed").textContent = `${avgSpeed.toFixed(
      1
    )} km/h`;

    const tLabel = document.getElementById("predictionTime");
    if (tLabel) {
      tLabel.textContent = finalLabel;
      tLabel.style.color = hoursOffset > 0 ? "#ffa726" : "#4fc3f7";
    }
    document.getElementById("sliderTime").textContent = `+${hoursOffset}h`;
  }

  setupEventListeners() {
    const slider = document.getElementById("predictionSlider");

    // Throttle the slider for performance
    let isUpdating = false;
    if (slider) {
      slider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        if (this.currentIndex !== val) {
          this.currentIndex = val;
          if (!isUpdating) {
            requestAnimationFrame(() => {
              this.updateMapState(this.currentIndex);
              isUpdating = false;
            });
            isUpdating = true;
          }
        }
      });
    }

    document
      .getElementById("backBtn")
      .addEventListener("click", () => (window.location.href = "/"));

    const toggleBtn = document.getElementById("toggleControlsBtn");
    const header = document.getElementById("controlHeader");
    const controls = document.getElementById("predictionControls");
    const toggleFn = () => {
      controls.classList.toggle("minimized");
      const isMin = controls.classList.contains("minimized");
      toggleBtn.textContent = isMin ? "▲" : "▼";
    };
    if (toggleBtn) toggleBtn.addEventListener("click", toggleFn);
    if (header)
      header.addEventListener("click", (e) => {
        if (e.target !== toggleBtn) toggleFn();
      });
  }

  addInteractivity() {
    this.map.on("click", "traffic-layer", (e) => {
      // READ FROM STATE (Dynamic), READ NAME FROM PROPERTIES (Static)
      const state = e.features[0].state;
      const props = e.features[0].properties;

      if (!state.current_speed) return;

      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(
          `
                    <div class="popup-content">
                        <h3>${props.road_name || "Road"}</h3>
                        <div class="popup-stats">
                            <div><strong>Speed:</strong> ${
                              state.current_speed
                            } km/h</div>
                            <div><strong>Time:</strong> ${new Date(
                              state.current_time
                            ).toLocaleTimeString()}</div>
                        </div>
                    </div>
                `
        )
        .addTo(this.map);
    });

    this.map.on(
      "mouseenter",
      "traffic-layer",
      () => (this.map.getCanvas().style.cursor = "pointer")
    );
    this.map.on(
      "mouseleave",
      "traffic-layer",
      () => (this.map.getCanvas().style.cursor = "")
    );
  }

  startClock() {
    const update = () => {
      const now = new Date();
      const timeString = now.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      });
      const dateString = now.toLocaleDateString("en-US", {
        weekday: "short",
        year: "numeric",
        month: "short",
        day: "numeric",
      });
      const el = document.getElementById("currentTime");
      if (el) el.textContent = `${dateString} ${timeString}`;
    };
    update();
    setInterval(update, 60000);
  }

  showErrorMessage(error) {
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) {
      overlay.style.display = "flex";
      overlay.innerHTML = `
                <div style="text-align:center; color: #ff6b6b;">
                    <div style="font-size:40px;">⚠️</div>
                    <h3>Data Load Failed</h3>
                    <p>${error.message}</p>
                    <button onclick="location.reload()" style="padding:8px 16px; margin-top:10px; border:none; border-radius:4px; background:#4fc3f7; cursor:pointer;">Retry</button>
                </div>
            `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new TrafficPredictions();
});
