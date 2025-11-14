class TrafficPredictions {
    constructor() {
        this.map = null;
        this.predictionData = null;
        this.currentHour = 0;
        this.predictionLayers = [];
        this.init();
    }

    async init() {
        await this.loadMap();
        await this.loadPredictionData();
        this.setupEventListeners();
        this.updateDisplay();
    }

    async loadMap() {
        // Initialize map
        this.map = new maplibregl.Map({
            container: 'map',
            style: {
                version: 8,
                sources: {
                    'osm-tiles': {
                        type: 'raster',
                        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                        tileSize: 256,
                        attribution: '© OpenStreetMap contributors'
                    }
                },
                layers: [{
                    id: 'osm-tiles',
                    type: 'raster',
                    source: 'osm-tiles',
                    minzoom: 0,
                    maxzoom: 19
                }]
            },
            center: [103.8198, 1.3521], // Singapore center
            zoom: 11
        });

        // Wait for map to load
        await new Promise(resolve => this.map.on('load', resolve));
    }

    async loadPredictionData() {
        try {
            const loadingOverlay = document.getElementById('loadingOverlay');
            loadingOverlay.style.display = 'flex';

            // Load prediction data from backend
            const response = await fetch('/api/predictions');
            if (!response.ok) {
                throw new Error('Failed to load prediction data');
            }

            this.predictionData = await response.json();
            this.createPredictionLayers();
            
            loadingOverlay.style.display = 'none';
            
        } catch (error) {
            console.error('Error loading prediction data:', error);
            document.getElementById('loadingOverlay').innerHTML = `
                <div style="text-align: center; color: #ff6b6b;">
                    <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                    <div>Failed to load prediction data</div>
                    <div style="font-size: 12px; margin-top: 10px;">${error.message}</div>
                </div>
            `;
        }
    }

    createPredictionLayers() {
        if (!this.predictionData || !this.map) return;

        // Remove existing prediction layers
        this.predictionLayers.forEach(layerId => {
            if (this.map.getLayer(layerId)) {
                this.map.removeLayer(layerId);
            }
            if (this.map.getSource(layerId)) {
                this.map.removeSource(layerId);
            }
        });
        this.predictionLayers = [];

        // Create GeoJSON features from prediction data
        const features = this.predictionData.map(segment => {
            return {
                type: 'Feature',
                geometry: {
                    type: 'LineString',
                    coordinates: [
                        [segment.start_lon, segment.start_lat],
                        [segment.end_lon, segment.end_lat]
                    ]
                },
                properties: {
                    id: segment.link_id,
                    roadName: segment.road_name,
                    predictedSpeed: segment.predicted_speed,
                    currentSpeed: segment.current_speed,
                    roadCategory: segment.road_category,
                    trafficCondition: segment.traffic_condition
                }
            };
        });

        // Add source and layer for predictions
        const sourceId = 'predictions-source';
        const layerId = 'predictions-layer';

        this.map.addSource(sourceId, {
            type: 'geojson',
            data: {
                type: 'FeatureCollection',
                features: features
            }
        });

        this.map.addLayer({
            id: layerId,
            type: 'line',
            source: sourceId,
            paint: {
                'line-color': [
                    'interpolate',
                    ['linear'],
                    ['get', 'predictedSpeed'],
                    0, '#ff0000',    // Red for 0 km/h
                    30, '#ffff00',   // Yellow for 30 km/h
                    60, '#00ff00'    // Green for 60+ km/h
                ],
                'line-width': [
                    'interpolate',
                    ['linear'],
                    ['zoom'],
                    10, 2,
                    15, 4,
                    18, 6
                ],
                'line-opacity': 0.8
            }
        });

        this.predictionLayers.push(sourceId, layerId);

        // Add click interaction
        this.map.on('click', layerId, (e) => {
            const properties = e.features[0].properties;
            new maplibregl.Popup()
                .setLngLat(e.lngLat)
                .setHTML(`
                    <div class="popup-content">
                        <h3>${properties.roadName || 'Unknown Road'}</h3>
                        <div class="popup-stats">
                            <div><strong>Predicted Speed:</strong> ${properties.predictedSpeed.toFixed(1)} km/h</div>
                            <div><strong>Current Speed:</strong> ${properties.currentSpeed.toFixed(1)} km/h</div>
                            <div><strong>Category:</strong> ${properties.roadCategory || 'N/A'}</div>
                            <div><strong>Condition:</strong> ${properties.trafficCondition || 'N/A'}</div>
                        </div>
                    </div>
                `)
                .addTo(this.map);
        });

        // Change cursor on hover
        this.map.on('mouseenter', layerId, () => {
            this.map.getCanvas().style.cursor = 'pointer';
        });

        this.map.on('mouseleave', layerId, () => {
            this.map.getCanvas().style.cursor = '';
        });

        this.updateStats();
    }

    setupEventListeners() {
        // Back button
        document.getElementById('backBtn').addEventListener('click', () => {
            window.location.href = '/';
        });

        // Prediction slider
        const slider = document.getElementById('predictionSlider');
        const sliderTime = document.getElementById('sliderTime');
        
        slider.addEventListener('input', (e) => {
            this.currentHour = parseInt(e.target.value);
            this.updatePredictionDisplay();
            sliderTime.textContent = `+${this.currentHour}h`;
        });

        // Update current time
        this.updateCurrentTime();
        setInterval(() => this.updateCurrentTime(), 60000); // Update every minute
    }

    updateCurrentTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
        const dateString = now.toLocaleDateString('en-US', {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        document.getElementById('currentTime').textContent = `${dateString} ${timeString}`;
    }

    updatePredictionDisplay() {
        const predictionTime = document.getElementById('predictionTime');
        
        if (this.currentHour === 0) {
            predictionTime.textContent = 'Now';
            predictionTime.style.color = '#4fc3f7';
        } else {
            const futureTime = new Date();
            futureTime.setHours(futureTime.getHours() + this.currentHour);
            const timeString = futureTime.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: true 
            });
            predictionTime.textContent = `+${this.currentHour}h (${timeString})`;
            predictionTime.style.color = '#ffa726';
        }

        // Simulate prediction changes (in real implementation, this would use actual prediction data)
        this.simulatePredictionChanges();
    }

    simulatePredictionChanges() {
        // This is a simulation - in real implementation, you would have actual prediction data for each hour
        if (this.map && this.map.getSource('predictions-source')) {
            // Update line colors based on simulated time-based changes
            this.map.setPaintProperty('predictions-layer', 'line-color', [
                'interpolate',
                ['linear'],
                ['get', 'predictedSpeed'],
                0, '#ff0000',
                30 - (this.currentHour * 0.5), '#ffff00', // Simulate traffic patterns
                60 - (this.currentHour * 1), '#00ff00'
            ]);
        }
    }

    updateStats() {
        if (!this.predictionData) return;

        const segmentCount = this.predictionData.length;
        const avgSpeed = this.predictionData.reduce((sum, segment) => sum + segment.predicted_speed, 0) / segmentCount;

        document.getElementById('segmentCount').textContent = segmentCount.toLocaleString();
        document.getElementById('avgSpeed').textContent = `${avgSpeed.toFixed(1)} km/h`;
    }

    updateDisplay() {
        this.updateCurrentTime();
        this.updatePredictionDisplay();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TrafficPredictions();
});

// Add popup styles
const style = document.createElement('style');
style.textContent = `
    .popup-content {
        padding: 8px;
        min-width: 200px;
    }
    
    .popup-content h3 {
        margin: 0 0 8px 0;
        color: #333;
        font-size: 14px;
        border-bottom: 1px solid #eee;
        padding-bottom: 4px;
    }
    
    .popup-stats div {
        margin: 4px 0;
        font-size: 12px;
        color: #666;
    }
    
    .popup-stats strong {
        color: #333;
    }
    
    .maplibregl-popup-content {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
`;
document.head.appendChild(style);