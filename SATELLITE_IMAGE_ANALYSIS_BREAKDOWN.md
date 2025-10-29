# Satellite Image Analysis Pipeline - Complete Breakdown

This document provides a comprehensive breakdown of the satellite image analysis system, from initial image capture through to final analyzed results.

## 🔄 Complete Workflow Overview

```

Pipeline Setup → Image Capture → Image Download → Image Processing →
Analysis → Anomaly Detection → Results Storage → Alert Generation

```

---

## 📋 Stage-by-Stage Breakdown

### **STAGE 1: Pipeline Configuration & Setup**

**Location**: `backend/monitoring/models.py` - `Pipeline` model

**What Happens**:

1. User creates a `Pipeline` record with:
   - Geographic coordinates (start_point, end_point, route)
   - Pipeline metadata (length, diameter, material, status)
   - Monitoring configuration (frequency, thresholds)

**Key Fields**:

- `start_point`: Starting geographic point (GIS PointField)
- `end_point`: Ending geographic point (GIS PointField)
- `route`: Full pipeline route as LineString
- `status`: 'active', 'maintenance', or 'inactive'
- `monitoring_config`: Configuration for monitoring frequency

**Files Involved**:

- `backend/monitoring/models.py` (Pipeline model)
- `backend/monitoring/views.py` (PipelineViewSet)

---

### **STAGE 2: Scheduled Image Capture Trigger**

**Location**: `backend/monitoring/tasks.py` - `fetch_satellite_imagery_periodic()`

**What Happens**:

1. **Celery Periodic Task** runs on schedule (e.g., hourly/daily)
2. System checks all `active` pipelines
3. For each pipeline, checks:
   - Last image capture date
   - Monitoring frequency (from config or default 24 hours)
   - If last image is older than frequency → triggers fetch

**Trigger Logic**:

```python
if last_image.image_date < timezone.now() - timedelta(hours=frequency_hours):
    fetch_pipeline_imagery.delay(str(pipeline.id))
```

**Files Involved**:

- `backend/monitoring/tasks.py` (fetch_satellite_imagery_periodic)
- `backend/pipeline_monitoring/celery.py` (Celery configuration)

---

### **STAGE 3: Satellite Image Fetching**

**Location**: `backend/scripts/satellite_data_fetcher.py`

#### **3.1 API Selection & Query**

**Classes**:

- `NASASatelliteFetcher`: NASA Earth Imagery API
- `LandsatFetcher`: USGS Landsat API (placeholder)
- `SentinelFetcher`: Copernicus Sentinel API (placeholder)

**What Happens**:

1. `SatelliteDataManager` receives pipeline_id, start_date, end_date
2. Retrieves `Pipeline` from database
3. Calculates pipeline bounds with buffer (~1km):
   ```python
   bounds = (min_lon, min_lat, max_lon, max_lat)
   center_lat, center_lon = calculate_center(bounds)
   ```
4. Calls appropriate fetcher based on sources list

#### **3.2 NASA API Request**

**Methods**: `NASASatelliteFetcher.get_assets()` and `get_imagery()`

**What Happens**:

1. **Asset Discovery**:

   - Queries NASA Assets API: `/planetary/earth/assets`
   - Parameters: lat, lon, begin_date, end_date, api_key
   - Returns list of available satellite images for date range

2. **Image Retrieval**:
   - For each asset, calls `/planetary/earth/imagery`
   - Gets actual image URL and metadata
   - Retrieves cloud score (quality indicator)

**Response Data**:

```python
{
    'url': 'https://...',  # Image URL
    'cloud_score': 0.15,   # Cloud cover percentage
    'date': '2024-01-15'   # Capture date
}
```

#### **3.3 Image Metadata Creation**

**Class**: `SatelliteImageData` (dataclass)

**What Happens**:
For each found image, creates `SatelliteImageData` object:

```python
SatelliteImageData(
    image_id="nasa_2024-01-15",
    satellite_name="Landsat",
    sensor="OLI_TIRS",
    image_date=datetime(...),
    bounds=(lon_min, lat_min, lon_max, lat_max),
    resolution_m=30.0,  # 30 meters per pixel
    image_url="https://...",
    cloud_cover=0.15,
    quality_score=0.85  # 1.0 - cloud_cover
)
```

**Files Involved**:

- `backend/scripts/satellite_data_fetcher.py`
  - `NASASatelliteFetcher` class
  - `SatelliteDataManager` class
  - `fetch_pipeline_imagery_task()` function

---

### **STAGE 4: Image Download & Storage**

**Location**: `backend/scripts/satellite_data_fetcher.py` - `download_and_process_image()`

#### **4.1 Database Record Creation**

**What Happens**:

1. Creates `SatelliteImage` record in database:
   ```python
   SatelliteImage.objects.create(
       pipeline=pipeline,
       image_date=image_data.image_date,
       satellite_name=image_data.satellite_name,
       sensor=image_data.sensor,
       resolution_m=image_data.resolution_m,
       bounds=Polygon.from_bbox(image_data.bounds),
       center_point=Point(center_lon, center_lat),
       source_api='nasa',
       api_image_id=image_data.image_id,
       processing_status='pending'
   )
   ```

#### **4.2 Image Download**

**What Happens**:

1. Downloads image from URL using `requests.get()`
2. Saves to: `MEDIA_ROOT/satellite_images/{uuid}.jpg`
3. File operations:
   - Opens image with PIL (Pillow)
   - Converts to RGB if needed
   - Resizes if too large (max 2048x2048)
   - Saves as JPEG with 85% quality

#### **4.3 Thumbnail Generation**

**What Happens**:

1. Creates thumbnail from original image
2. Size: 300x300 pixels
3. Saves to: `MEDIA_ROOT/thumbnails/thumb_{uuid}.jpg`
4. Quality: 80%

#### **4.4 Status Update**

**What Happens**:

1. Updates `SatelliteImage.processing_status`:
   - `'completed'` if download successful
   - `'failed'` if download fails
2. Updates database with file paths

**Files Involved**:

- `backend/scripts/satellite_data_fetcher.py`
  - `download_and_process_image()` method
  - `create_thumbnail()` method
  - `fetch_pipeline_imagery_task()` function
- `backend/monitoring/models.py` - `SatelliteImage` model

---

### **STAGE 5: Image Preprocessing**

**Location**: `backend/scripts/image_analyzer.py` - `ImagePreprocessor` class

#### **5.1 Image Loading**

**Method**: `ImagePreprocessor.load_image()`

**What Happens**:

1. Uses OpenCV (`cv2.imread()`) to load image
2. Converts from BGR to RGB color space
3. Returns numpy array representation

#### **5.2 Image Enhancement**

**Method**: `ImagePreprocessor.enhance_image()`

**What Happens**:

1. **Color Space Conversion**: RGB → LAB color space
2. **CLAHE Application**:
   - Applies Contrast Limited Adaptive Histogram Equalization
   - Only to L (Lightness) channel
   - Parameters: clipLimit=2.0, tileGridSize=(8,8)
3. **Channel Merge**: Combines enhanced L with original A and B
4. **Conversion Back**: LAB → RGB

**Purpose**: Improves contrast for better anomaly detection

#### **5.3 Additional Preprocessing**

- **Normalization**: Converts pixel values to 0-1 range (if needed)
- **Resizing**: Resizes to target size (default 512x512) for consistent processing

**Files Involved**:

- `backend/scripts/image_analyzer.py`
  - `ImagePreprocessor` class (lines 44-85)

---

### **STAGE 6: Image Analysis**

**Location**: `backend/scripts/image_analyzer.py` - `ImageAnalyzer` class

**Task Trigger**: `backend/monitoring/tasks.py` - `analyze_satellite_image()`

#### **6.1 Analysis Orchestration**

**Method**: `ImageAnalyzer.analyze_image()`

**What Happens**:

1. Loads and enhances image (Stage 5)
2. Optionally loads reference image for comparison
3. Runs multiple detection algorithms based on `analysis_types`:
   - `'leak_detection'` → `LeakDetector`
   - `'oil_spill'` → `OilSpillDetector`
   - `'vandalism'` → `VandalismDetector`
   - `'anomaly'` → `AnomalyDetector`

#### **6.2 Leak Detection**

**Class**: `LeakDetector`

**Algorithm**:

1. **Color Space Conversion**: RGB → HSV
2. **Color Masking**:
   - Defines HSV range for leak colors (dark spots, discoloration)
   - Range: `lower=[0,50,50]`, `upper=[20,255,255]`
3. **Morphological Operations**:
   - Closing: Fills small gaps
   - Opening: Removes noise
4. **Contour Detection**:
   - Finds contours using `cv2.findContours()`
   - Filters by minimum area (100 pixels)
5. **Confidence Calculation**:
   - Based on area size and shape
   - Confidence = min(area / 1000, 1.0)
6. **Severity Classification**:
   - `critical`: area > 500, confidence > 0.8
   - `high`: area > 200, confidence > 0.6
   - `medium`: area > 100, confidence > 0.4
   - `low`: otherwise

**Output**: `AnalysisResult` object with:

- Type: 'leak_detection'
- Confidence score (0-1)
- Severity level
- Bounding box coordinates
- Location (center point)
- Raw data (area, contour points, aspect ratio)

#### **6.3 Oil Spill Detection**

**Class**: `OilSpillDetector`

**Algorithm**:

1. **Multi-Color Space Analysis**:
   - Converts to HSV and LAB
2. **Dark Spot Detection**:
   - Uses adaptive threshold on grayscale
   - Finds dark regions (oil appears dark)
3. **Color-Based Masking**:
   - Creates mask for oil-like colors
   - Range: HSV [100-130, 50-255, 20-100] (bluish-black)
   - Also detects very dark areas (V < 50)
4. **Mask Combination**: Combines dark spots and color masks
5. **Contour Analysis**:
   - Finds contours
   - Calculates circularity (oil spills are circular)
   - Calculates aspect ratio
6. **Confidence Calculation**:
   - Based on circularity and aspect ratio
   - Formula: `(circularity * 0.6) + (aspect_ratio_factor * 0.4)`
7. **Severity Classification**: Similar to leak detection

**Output**: `AnalysisResult` with oil spill characteristics

#### **6.4 Vandalism Detection**

**Class**: `VandalismDetector`

**Algorithm**:

1. **Reference Comparison** (if available):
   - Calculates absolute difference between current and reference images
   - Applies threshold to find significant changes
   - Identifies new objects/structures
2. **Suspicious Pattern Detection** (if no reference):
   - Detects unusual geometric patterns
   - Identifies construction equipment
   - Detects unauthorized structures
3. **Change Analysis**:
   - Cluster analysis of changed pixels
   - Size and shape filtering
   - Temporal change validation

**Output**: `AnalysisResult` with vandalism/activity indicators

#### **6.5 General Anomaly Detection**

**Class**: `AnomalyDetector`

**Algorithm**:

1. **Feature Extraction**:
   - Extracts statistical features (mean, std, variance)
   - Color histogram features
   - Texture features (using GLCM or similar)
   - Edge density features
2. **Machine Learning Detection**:
   - Uses `IsolationForest` (if trained)
   - PCA for dimensionality reduction
   - StandardScaler for normalization
3. **Fallback Method** (if not trained):
   - Statistical outlier detection
   - Z-score based anomalies
   - Local stark deviations

**Output**: `AnalysisResult` with anomaly detection results

**Files Involved**:

- `backend/scripts/image_analyzer.py`
  - `ImageAnalyzer` class (main orchestrator)
  - `LeakDetector` class
  - `OilSpillDetector` class
  - `VandalismDetector` class
  - `AnomalyDetector` class
- `backend/monitoring/tasks.py`
  - `analyze_satellite_image()` task
  - `analyze_images_periodic()` task

---

### **STAGE 7: Advanced Anomaly Detection (Time Series & Spatial)**

**Location**: `backend/scripts/anomaly_detector.py`

**Task Trigger**: `backend/monitoring/tasks.py` - `detect_anomalies()`

#### **7.1 Time Series Anomaly Detection**

**Class**: `TimeSeriesAnomalyDetector`

**What Happens**:

1. **Statistical Anomaly Detection**:
   - Z-score calculation
   - IQR (Interquartile Range) method
   - Flags values outside normal distribution
2. **Trend Anomaly Detection**:
   - Moving average calculation
   - Detects sudden trend changes
   - Identifies gradual degradation patterns
3. **Seasonal Anomaly Detection**:
   - Decomposes time series (trend, seasonal, residual)
   - Detects anomalies in seasonal patterns
   - Accounts for expected variations
4. **Anomaly Merging**:
   - Removes duplicate detections
   - Groups nearby anomalies
   - Prioritizes significant anomalies

#### **7.2 Spatial Clustering**

**Class**: `SpatialAnomalyDetector`

**What Happens**:

1. **DBSCAN Clustering**:
   - Groups spatially close anomalies
   - Identifies anomaly clusters
   - Filters noise points
2. **Spatial Pattern Analysis**:
   - Detects linear patterns (potential pipeline issues)
   - Identifies circular patterns (spill patterns)
   - Analyzes spatial distribution

#### **7.3 Ensemble Detection**

**Class**: `EnsembleAnomalyDetector`

**What Happens**:

1. Combines multiple detection methods:
   - IsolationForest
   - Local Outlier Factor (LOF)
   - One-Class SVM
   - Statistical methods
2. **Voting Mechanism**:
   - Majority voting
   - Weighted voting based on confidence
   - Consensus-based final decision

**Files Involved**:

- `backend/scripts/anomaly_detector.py`
  - `TimeSeriesAnomalyDetector`
  - `SpatialAnomalyDetector`
  - `EnsembleAnomalyDetector`
  - `AnomalyDetectionManager`
- `backend/monitoring/tasks.py`
  - `detect_anomalies()` task
  - `detect_anomalies_periodic()` task

---

### **STAGE 8: Results Storage**

**Location**: `backend/scripts/image_analyzer.py` - `analyze_satellite_image_task()`

#### **8.1 Database Record Creation**

**Model**: `AnalysisResult` (`backend/monitoring/models.py`)

**What Happens**:
For each detected anomaly:

```python
AnalysisResult.objects.create(
    satellite_image=satellite_image,
    analysis_type='leak_detection',  # or oil_spill, vandalism, etc.
    confidence_score=0.85,
    severity='high',
    detected_location=Point(lon, lat),
    affected_area=Polygon(...),  # Optional
    description="Potential pipeline leak detected...",
    raw_data={...},  # JSON with detailed metrics
    status='pending'  # Waiting for review
)
```

**Fields Stored**:

- `analysis_type`: Type of detection
- `confidence_score`: 0.0 to 1.0
- `severity`: 'low', 'medium', 'high', 'critical'
- `detected_location`: Geographic point (GIS)
- `affected_area`: Polygon of affected region (GIS)
- `description`: Human-readable description
- `raw_data`: JSON with detailed analysis metrics
- `status`: 'pending', 'verified', 'false_positive', etc.

#### **8.2 Image Status Update**

**What Happens**:

- Updates `SatelliteImage` to mark as analyzed
- Links analysis results via ForeignKey relationship

**Files Involved**:

- `backend/monitoring/models.py` - `AnalysisResult` model
- `backend/scripts/image_analyzer.py` - `analyze_satellite_image_task()`

---

### **STAGE 9: Alert Generation**

**Location**: `backend/monitoring/models.py` - `MonitoringAlert` model

#### **9.1 Alert Creation Logic**

**Trigger**: When `AnalysisResult` is created with:

- `severity` = 'high' or 'critical'
- `confidence_score` > threshold (typically 0.7)

**What Happens**:

```python
MonitoringAlert.objects.create(
    analysis_result=analysis_result,
    alert_type='leak_detected',  # Maps to analysis_type
    priority='high',  # Based on severity
    message="High confidence leak detected...",
    is_resolved=False,
    sent_at=None  # Will be set when notification sent
)
```

#### **9.2 Alert Prioritization**

**Priority Levels**:

- `critical`: severity='critical' AND confidence > 0.8
- `high`: severity='high' OR (severity='medium' AND confidence > 0.7)
- `medium`: severity='medium'
- `low`: severity='low'

#### **9.3 Notification Sending**

**Task**: `send_alert_notifications()` in `tasks.py`

**What Happens**:

1. Finds all unresolved alerts not yet sent
2. For each alert:
   - Updates `sent_at` timestamp
   - Sends notification (email/SMS - TODO: implementation)
   - Logs notification status

**Files Involved**:

- `backend/monitoring/models.py` - `MonitoringAlert` model
- `backend/monitoring/tasks.py` - `send_alert_notifications()`

---

### **STAGE 10: Results Visualization & Review**

**Location**: Frontend (`frontend/src/pages/AnalysisResults.tsx`) and API (`backend/monitoring/views.py`)

#### **10.1 API Endpoints**

**ViewSet**: `AnalysisResultViewSet`

**Available Actions**:

- `GET /api/analysis-results/` - List all results
- `GET /api/analysis-results/{id}/` - Get specific result
- `PATCH /api/analysis-results/{id}/` - Update status (verify, dismiss)
- `GET /api/analysis-results/?pipeline={id}` - Filter by pipeline
- `GET /api/analysis-results/?analysis_type={type}` - Filter by type

#### **10.2 Status Management**

**Status Workflow**:

```
pending → verified/alert_created → resolved/dismissed
         ↓
    false_positive → dismissed
```

**Actions Available**:

- **Verify**: Mark as verified (human confirmation)
- **Create Alert**: Manually create alert from result
- **Mark False Positive**: Dismiss as false detection
- **Resolve**: Mark issue as resolved
- **Dismiss**: Remove from active monitoring

#### **10.3 Frontend Display**

**Components**:

- `AnalysisResults.tsx`: Main results listing page
- Shows: confidence score, severity, location on map
- Filters: by pipeline, type, severity, date range
- Actions: verify, dismiss, view details

**Files Involved**:

- `backend/monitoring/views.py` - `AnalysisResultViewSet`
- `backend/monitoring/serializers.py` - `AnalysisResultSerializer`
- `frontend/src/pages/AnalysisResults.tsx`

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────┐
│ Pipeline Model  │
│ (Configuration) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Periodic Celery Task    │
│ (fetch_satellite_...)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ SatelliteDataManager    │
│ - Calculate bounds      │
│ - Query NASA API        │
│ - Get image URLs        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Download & Store        │
│ - Download image        │
│ - Create thumbnail      │
│ - Save to database      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ImagePreprocessor       │
│ - Load image            │
│ - Enhance (CLAHE)       │
│ - Normalize             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ImageAnalyzer           │
│ ├─ LeakDetector         │
│ ├─ OilSpillDetector     │
│ ├─ VandalismDetector    │
│ └─ AnomalyDetector      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ AnalysisResult Model    │
│ - Store results         │
│ - Link to image         │
│ - Geographic data       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Anomaly Detection       │
│ (Time Series & Spatial) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Alert Generation        │
│ (if severity high)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Frontend Display        │
│ - Results listing       │
│ - Map visualization     │
│ - Status management     │
└─────────────────────────┘
```

---

## 📊 Key Technologies & Libraries

### **Image Processing**:

- **OpenCV (cv2)**: Image loading, color space conversion, contour detection, morphological operations
- **PIL/Pillow**: Image format conversion, resizing, thumbnail creation
- **NumPy**: Array operations, mathematical calculations
- **Rasterio**: Geospatial raster data handling (for future use)

### **Machine Learning**:

- **scikit-learn**:
  - `IsolationForest`: Anomaly detection
  - `DBSCAN`: Spatial clustering
  - `PCA`: Dimensionality reduction
  - `StandardScaler`: Feature normalization
- **Pandas**: Time series data handling

### **Geospatial**:

- **PostGIS**: Geographic data storage (Polygon, Point, LineString)
- **Django GIS**: Geographic model fields and queries

### **Task Processing**:

- **Celery**: Asynchronous task execution
- **Redis**: Message broker and result backend

### **API Integration**:

- **NASA Earth Imagery API**: Satellite image retrieval
- **requests**: HTTP requests for API calls

---

## 🎯 Key Metrics Tracked

### **Image Quality Metrics**:

- Cloud cover percentage
- Resolution (meters per pixel)
- Satellite source
- Capture date/time

### **Analysis Metrics**:

- Confidence score (0.0 - 1.0)
- Severity level (low, medium, high, critical)
- Detected area (pixels or square meters)
- Location (latitude, longitude)
- Affected area polygon

### **Performance Metrics**:

- Processing time per image
- Detection accuracy
- False positive rate
- Alert response time

---

## 🔧 Configuration Points

### **Monitoring Frequency**:

- Default: 24 hours
- Configurable per pipeline
- Controlled by Celery beat schedule

### **Detection Thresholds**:

- Leak detection: `leak_threshold = 0.7`
- Oil spill: `spill_threshold = 0.6`
- Vandalism: `vandalism_threshold = 0.5`
- Minimum areas: 100-200 pixels (configurable)

### **Alert Thresholds**:

- High confidence: > 0.7
- Critical: severity='critical' AND confidence > 0.8

---

## 📝 Summary

The complete pipeline transforms raw satellite imagery into actionable insights through:

1. **Automated Scheduling**: Periodic checks and image fetching
2. **Multi-Source Integration**: NASA, Landsat, Sentinel APIs
3. **Advanced Processing**: Image enhancement and normalization
4. **Multiple Detection Algorithms**: Leaks, spills, vandalism, general anomalies
5. **Time Series Analysis**: Historical pattern detection
6. **Geographic Mapping**: Spatial analysis and visualization
7. **Alert System**: Automated notifications for critical findings
8. **Review Workflow**: Human verification and status management

The system is designed to be scalable, extensible, and production-ready with proper error handling, logging, and database optimization.
