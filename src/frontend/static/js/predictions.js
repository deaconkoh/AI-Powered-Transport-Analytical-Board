class TrafficPredictions {
    constructor() {
        this.map = null;
        this.predictionData = null;
        this.predictionData24h = null; // Store 24-hour predictions
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

            console.log('🔄 Loading prediction data from API...');
            
            // Load prediction data from backend
            const response = await fetch('/api/predictions');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('📊 Raw API response:', data);
            
            // Extract the predictions array from the response
            if (data.predictions && Array.isArray(data.predictions)) {
                this.predictionData = data.predictions;
                console.log(`✅ Loaded ${this.predictionData.length} predictions from 'predictions' array`);
                
                // Generate 24-hour predictions based on time patterns
                this.generate24hPredictions();
            } else if (data.heatmap_data && Array.isArray(data.heatmap_data)) {
                // Fallback for mock data structure
                this.predictionData = data.heatmap_data;
                console.log(`✅ Loaded ${this.predictionData.length} predictions from 'heatmap_data' array`);
                this.generate24hPredictions();
            } else if (Array.isArray(data)) {
                // If it's already an array, use it directly
                this.predictionData = data;
                console.log(`✅ Loaded ${this.predictionData.length} predictions from direct array`);
                this.generate24hPredictions();
            } else {
                console.warn('⚠️ Unexpected response format, using empty array');
                this.predictionData = [];
                this.predictionData24h = [];
            }
            
            if (this.predictionData.length === 0) {
                console.warn('⚠️ No prediction data available');
                this.showNoDataMessage();
            } else {
                this.createPredictionLayers();
                loadingOverlay.style.display = 'none';
            }
            
        } catch (error) {
            console.error('❌ Error loading prediction data:', error);
            this.showErrorMessage(error);
        }
    }

    generate24hPredictions() {
        if (!this.predictionData || this.predictionData.length === 0) return;

        console.log('🔄 Generating 24-hour predictions based on time patterns...');
        
        this.predictionData24h = [];
        
        // Generate predictions for each hour (0-23)
        for (let hour = 0; hour < 24; hour++) {
            const hourPredictions = this.predictionData.map(segment => {
                // Create a copy of the segment
                const newSegment = {...segment};
                
                // Calculate time-based speed adjustment
                const timeAdjustment = this.calculateTimeAdjustment(hour, newSegment);
                
                // Apply time-based adjustment to predicted speed
                newSegment.predicted_speed = Math.max(5, Math.min(100, 
                    newSegment.predicted_speed * timeAdjustment
                ));
                
                // Update traffic condition based on adjusted speed
                newSegment.traffic_condition = this.getTrafficCondition(newSegment.predicted_speed);
                
                // Add hour information
                newSegment.hour = hour;
                newSegment.display_hour = hour === 0 ? 'Now' : `+${hour}h`;
                
                return newSegment;
            });
            
            this.predictionData24h.push(hourPredictions);
        }
        
        console.log(`✅ Generated 24 hours of predictions (${this.predictionData24h.length} hours)`);
    }

    calculateTimeAdjustment(hour, segment) {
        // Base adjustment factors for different times of day
        // These simulate typical traffic patterns
        let baseAdjustment = 1.0;
        
        // Morning rush hour (7-9 AM)
        if (hour >= 7 && hour <= 9) {
            baseAdjustment = 0.6; // Heavy traffic
        }
        // Evening rush hour (5-7 PM)
        else if (hour >= 17 && hour <= 19) {
            baseAdjustment = 0.7; // Heavy traffic
        }
        // Lunch time (12-1 PM)
        else if (hour >= 12 && hour <= 13) {
            baseAdjustment = 0.8; // Moderate traffic
        }
        // Late night (10 PM - 5 AM)
        else if (hour >= 22 || hour <= 5) {
            baseAdjustment = 1.4; // Light traffic
        }
        // Mid-day (10 AM - 4 PM)
        else if (hour >= 10 && hour <= 16) {
            baseAdjustment = 1.1; // Light traffic
        }
        
        // Add some randomness and road-type specific adjustments
        const roadTypeFactor = this.getRoadTypeFactor(segment.road_category);
        const randomFactor = 0.9 + (Math.random() * 0.2); // 0.9 to 1.1
        
        return baseAdjustment * roadTypeFactor * randomFactor;
    }

    getRoadTypeFactor(roadCategory) {
        // Different road types have different traffic patterns
        switch (roadCategory?.toLowerCase()) {
            case 'expressway':
                return 1.2; // Expressways handle traffic better
            case 'highway':
                return 1.3;
            case 'arterial':
                return 1.0;
            case 'residential':
                return 0.8; // Residential streets more sensitive to traffic
            default:
                return 1.0;
        }
    }

    getTrafficCondition(speed) {
        if (speed >= 60) return "Fluid";
        if (speed >= 40) return "Moderate";
        if (speed >= 20) return "Slow";
        return "Congested";
    }

    showNoDataMessage() {
        document.getElementById('loadingOverlay').innerHTML = `
            <div style="text-align: center; color: #ffa726;">
                <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
                <div>No prediction data available</div>
                <div style="font-size: 12px; margin-top: 10px;">The prediction model may still be loading</div>
                <button onclick="location.reload()" style="margin-top: 15px; padding: 8px 16px; background: #4fc3f7; border: none; border-radius: 4px; color: white; cursor: pointer;">
                    Retry
                </button>
            </div>
        `;
    }

    showErrorMessage(error) {
        document.getElementById('loadingOverlay').innerHTML = `
            <div style="text-align: center; color: #ff6b6b;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                <div>Failed to load prediction data</div>
                <div style="font-size: 12px; margin-top: 10px;">${error.message}</div>
                <button onclick="location.reload()" style="margin-top: 15px; padding: 8px 16px; background: #4fc3f7; border: none; border-radius: 4px; color: white; cursor: pointer;">
                    Retry
                </button>
            </div>
        `;
    }

    createPredictionLayers() {
        if (!this.predictionData24h || this.predictionData24h.length === 0) {
            console.warn('No 24-hour prediction data available');
            return;
        }

        const currentData = this.predictionData24h[this.currentHour];
        
        if (!currentData || currentData.length === 0) {
            console.warn('No prediction data for current hour');
            return;
        }

        if (!this.map) {
            console.error('Map not initialized');
            return;
        }

        console.log(`🔄 Creating prediction layers for hour ${this.currentHour}...`);

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
        const features = currentData.map(segment => {
            // Handle different field naming conventions
            const startLon = segment.start_lon || segment.StartLongitude || 103.8198;
            const startLat = segment.start_lat || segment.StartLatitude || 1.3521;
            const endLon = segment.end_lon || segment.EndLongitude || 103.8198;
            const endLat = segment.end_lat || segment.EndLatitude || 1.3521;
            const predictedSpeed = segment.predicted_speed || segment.predictedSpeed || 50;
            const currentSpeed = segment.current_speed || segment.currentSpeed || 50;
            const roadName = segment.road_name || segment.roadName || 'Unknown Road';
            const roadCategory = segment.road_category || segment.roadCategory || 'arterial';
            const trafficCondition = segment.traffic_condition || segment.trafficCondition || 'Moderate';

            return {
                type: 'Feature',
                geometry: {
                    type: 'LineString',
                    coordinates: [
                        [startLon, startLat],
                        [endLon, endLat]
                    ]
                },
                properties: {
                    id: segment.link_id || segment.id || `road_${Math.random().toString(36).substr(2, 9)}`,
                    roadName: roadName,
                    predictedSpeed: predictedSpeed,
                    currentSpeed: currentSpeed,
                    roadCategory: roadCategory,
                    trafficCondition: trafficCondition,
                    hour: segment.hour || this.currentHour,
                    displayHour: segment.display_hour || `+${this.currentHour}h`
                }
            };
        });

        console.log(`✅ Created ${features.length} GeoJSON features for hour ${this.currentHour}`);

        // Add source and layer for predictions
        const sourceId = 'predictions-source';
        const layerId = 'predictions-layer';

        try {
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
                        20, '#ff4444',   // Dark red for 20 km/h
                        40, '#ffff00',   // Yellow for 40 km/h
                        60, '#00ff00',   // Green for 60 km/h
                        80, '#00cc00'    // Dark green for 80+ km/h
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
            console.log('✅ Prediction layers added to map');

            // Add click interaction
            this.map.on('click', layerId, (e) => {
                const properties = e.features[0].properties;
                const hourDisplay = properties.displayHour || `+${this.currentHour}h`;
                
                new maplibregl.Popup()
                    .setLngLat(e.lngLat)
                    .setHTML(`
                        <div class="popup-content">
                            <h3>${properties.roadName || 'Unknown Road'}</h3>
                            <div class="popup-stats">
                                <div><strong>Time:</strong> ${hourDisplay}</div>
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

        } catch (error) {
            console.error('❌ Error adding layers to map:', error);
        }
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
            this.updateMapForCurrentHour();
            sliderTime.textContent = `+${this.currentHour}h`;
        });

        // Update current time
        this.updateCurrentTime();
        setInterval(() => this.updateCurrentTime(), 60000); // Update every minute
    }

    updateMapForCurrentHour() {
        if (!this.predictionData24h || this.predictionData24h.length === 0) {
            console.warn('No 24-hour data available for update');
            return;
        }

        const currentData = this.predictionData24h[this.currentHour];
        
        if (!currentData || currentData.length === 0) {
            console.warn('No data available for current hour:', this.currentHour);
            return;
        }

        // Update the GeoJSON source with new data
        const features = currentData.map(segment => {
            const startLon = segment.start_lon || segment.StartLongitude || 103.8198;
            const startLat = segment.start_lat || segment.StartLatitude || 1.3521;
            const endLon = segment.end_lon || segment.EndLongitude || 103.8198;
            const endLat = segment.end_lat || segment.EndLatitude || 1.3521;
            const predictedSpeed = segment.predicted_speed || segment.predictedSpeed || 50;

            return {
                type: 'Feature',
                geometry: {
                    type: 'LineString',
                    coordinates: [
                        [startLon, startLat],
                        [endLon, endLat]
                    ]
                },
                properties: {
                    id: segment.link_id || segment.id || `road_${Math.random().toString(36).substr(2, 9)}`,
                    roadName: segment.road_name || segment.roadName || 'Unknown Road',
                    predictedSpeed: predictedSpeed,
                    currentSpeed: segment.current_speed || segment.currentSpeed || 50,
                    roadCategory: segment.road_category || segment.roadCategory || 'arterial',
                    trafficCondition: segment.traffic_condition || segment.trafficCondition || 'Moderate',
                    hour: segment.hour || this.currentHour,
                    displayHour: segment.display_hour || `+${this.currentHour}h`
                }
            };
        });

        // Update the map source
        if (this.map && this.map.getSource('predictions-source')) {
            this.map.getSource('predictions-source').setData({
                type: 'FeatureCollection',
                features: features
            });
            console.log(`✅ Updated map for hour ${this.currentHour}`);
        }

        this.updateStats();
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
    }

    updateStats() {
        if (!this.predictionData24h || this.predictionData24h.length === 0) {
            document.getElementById('segmentCount').textContent = '0';
            document.getElementById('avgSpeed').textContent = '0 km/h';
            return;
        }

        const currentData = this.predictionData24h[this.currentHour];
        
        if (!currentData || currentData.length === 0) {
            document.getElementById('segmentCount').textContent = '0';
            document.getElementById('avgSpeed').textContent = '0 km/h';
            return;
        }

        const segmentCount = currentData.length;
        const avgSpeed = currentData.reduce((sum, segment) => {
            const speed = segment.predicted_speed || segment.predictedSpeed || 0;
            return sum + speed;
        }, 0) / segmentCount;

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