# Pipeline Monitoring System - Backend

A comprehensive Django REST API backend for monitoring pipeline infrastructure using satellite imagery and machine learning analysis.

## Features

### Core Functionality

- **Pipeline Management**: CRUD operations for pipeline infrastructure
- **Satellite Imagery Integration**: NASA API integration for satellite data
- **Image Analysis**: Computer vision and ML-based analysis for:
  - Pipeline leak detection
  - Oil spill monitoring
  - Vandalism detection
  - General anomaly detection
- **Real-time Monitoring**: Celery-based background tasks
- **Alert System**: Automated alert generation and notification
- **Geospatial Support**: PostGIS integration for geographic data

### API Endpoints

#### Authentication

- `POST /api/auth/users/` - User registration
- `POST /api/auth/jwt/create/` - Login
- `POST /api/auth/jwt/refresh/` - Refresh token
- `POST /api/auth/jwt/logout/` - Logout

#### Pipelines

- `GET /api/pipelines/` - List all pipelines
- `POST /api/pipelines/` - Create new pipeline
- `GET /api/pipelines/{id}/` - Get pipeline details
- `PUT /api/pipelines/{id}/` - Update pipeline
- `DELETE /api/pipelines/{id}/` - Delete pipeline
- `GET /api/pipelines/{id}/segments/` - Get pipeline segments
- `GET /api/pipelines/{id}/recent-images/` - Get recent satellite images
- `GET /api/pipelines/{id}/analysis-summary/` - Get analysis summary

#### Satellite Images

- `GET /api/satellite-images/` - List satellite images
- `POST /api/satellite-images/` - Upload new image
- `GET /api/satellite-images/{id}/` - Get image details
- `POST /api/satellite-images/{id}/analyze/` - Trigger image analysis
- `GET /api/satellite-images/{id}/analysis-results/` - Get analysis results

#### Analysis Results

- `GET /api/analysis-results/` - List analysis results
- `GET /api/analysis-results/{id}/` - Get analysis details
- `POST /api/analysis-results/{id}/verify/` - Verify analysis result
- `POST /api/analysis-results/{id}/mark-false-positive/` - Mark as false positive
- `POST /api/analysis-results/{id}/create-alert/` - Create alert

#### Monitoring Alerts

- `GET /api/alerts/` - List alerts
- `GET /api/alerts/active/` - Get active alerts
- `POST /api/alerts/{id}/resolve/` - Resolve alert

#### Dashboard

- `GET /api/dashboard/stats/` - Get dashboard statistics
- `GET /api/dashboard/recent-activity/` - Get recent activity
- `GET /api/dashboard/pipeline-health/` - Get pipeline health overview

## Technology Stack

- **Framework**: Django 4.2.7 + Django REST Framework
- **Database**: PostgreSQL with PostGIS extension
- **Authentication**: Djoser + JWT
- **Task Queue**: Celery + Redis
- **Geospatial**: GDAL, GeoPandas, Rasterio
- **ML/AI**: scikit-learn, OpenCV, NumPy
- **API Documentation**: drf-yasg (Swagger)
- **Containerization**: Docker + Docker Compose

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL with PostGIS
- Redis
- GDAL libraries

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd pipeline-monitoring/backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database**

   ```bash
   # Create PostgreSQL database with PostGIS
   createdb pipeline_monitoring
   psql pipeline_monitoring -c "CREATE EXTENSION postgis;"
   ```

6. **Run migrations**

   ```bash
   python manage.py migrate
   ```

7. **Create superuser**

   ```bash
   python manage.py createsuperuser
   ```

8. **Start development server**
   ```bash
   python manage.py runserver
   ```

### Docker Development

1. **Start services**

   ```bash
   docker-compose up -d
   ```

2. **Run migrations**

   ```bash
   docker-compose exec web python manage.py migrate
   ```

3. **Create superuser**

   ```bash
   docker-compose exec web python manage.py createsuperuser
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Admin: http://localhost:8000/admin
   - API Docs: http://localhost:8000/swagger/

## Configuration

### Environment Variables

| Variable            | Description       | Default                    |
| ------------------- | ----------------- | -------------------------- |
| `SECRET_KEY`        | Django secret key | Required                   |
| `DEBUG`             | Debug mode        | `True`                     |
| `DB_NAME`           | Database name     | `pipeline_monitoring`      |
| `DB_USER`           | Database user     | `postgres`                 |
| `DB_PASSWORD`       | Database password | `password`                 |
| `DB_HOST`           | Database host     | `localhost`                |
| `NASA_API_KEY`      | NASA API key      | Required                   |
| `CELERY_BROKER_URL` | Celery broker URL | `redis://localhost:6379/0` |

### NASA API Setup

1. Get API key from [NASA API Portal](https://api.nasa.gov/)
2. Add to environment variables
3. Configure in monitoring settings

## Usage

### Creating a Pipeline

```python
# Example pipeline data
pipeline_data = {
    "name": "South-South Nigeria Pipeline",
    "description": "30km pipeline in South-South region",
    "length_km": 30.0,
    "diameter_mm": 500.0,
    "material": "steel",
    "status": "active",
    "start_lat": 5.5,
    "start_lon": 6.0,
    "end_lat": 5.7,
    "end_lon": 6.2
}
```

### Fetching Satellite Imagery

```python
# Trigger satellite imagery fetch
from monitoring.tasks import fetch_pipeline_imagery

fetch_pipeline_imagery.delay(
    pipeline_id="your-pipeline-id",
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

### Running Analysis

```python
# Trigger image analysis
from monitoring.tasks import analyze_satellite_image

analyze_satellite_image.delay(
    image_id="your-image-id",
    analysis_types=['leak_detection', 'oil_spill', 'vandalism']
)
```

## API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/swagger/
- ReDoc: http://localhost:8000/redoc/

## Background Tasks

The system uses Celery for background processing:

### Periodic Tasks

- **Satellite Imagery Fetch**: Every 24 hours
- **Image Analysis**: Every 6 hours
- **Anomaly Detection**: Every 12 hours
- **Data Cleanup**: Every 7 days

### Manual Tasks

- `fetch_pipeline_imagery`: Fetch imagery for specific pipeline
- `analyze_satellite_image`: Analyze specific image
- `detect_anomalies`: Detect anomalies for pipeline
- `send_alert_notifications`: Send alert notifications

## Monitoring and Logging

- **Logs**: Stored in `logs/django.log`
- **Health Checks**: Built-in health check endpoints
- **Metrics**: Dashboard statistics endpoint

## Security

- JWT-based authentication
- CORS configuration
- Input validation and sanitization
- SQL injection protection
- XSS protection

## Performance

- Database query optimization
- Caching with Redis
- Image processing optimization
- Background task processing
- API pagination

## Troubleshooting

### Common Issues

1. **GDAL Installation Issues**

   ```bash
   # Install GDAL system dependencies
   sudo apt-get install gdal-bin libgdal-dev
   ```

2. **PostGIS Extension Missing**

   ```sql
   CREATE EXTENSION postgis;
   ```

3. **Celery Worker Not Starting**

   ```bash
   # Check Redis connection
   redis-cli ping
   ```

4. **NASA API Rate Limits**
   - Check API key validity
   - Implement rate limiting
   - Use caching for repeated requests

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

MIT License - see LICENSE file for details.
