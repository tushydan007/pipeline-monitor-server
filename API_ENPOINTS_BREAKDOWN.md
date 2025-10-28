# Pipeline Monitoring API Endpoints Breakdown

## Table of Contents

1. [Overview](#overview)
2. [How Endpoints Communicate](#how-endpoints-communicate)
3. [Complete API Endpoints List](#complete-api-endpoints-list)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Authentication Flow](#authentication-flow)
6. [Background Tasks](#background-tasks)

---

## Overview

The Pipeline Monitoring System is a Django REST Framework-based application that monitors pipeline infrastructure using satellite imagery. The API uses JWT authentication via Djoser and communicates with a React frontend.

**Base URL**: `http://localhost:8000/api`

---

## How Endpoints Communicate

### Core Workflow

1. **User Authentication** → Frontend logs in via `/api/auth/jwt/create/`
2. **Access Protected Resources** → All subsequent requests include JWT token in Authorization header
3. **Data Retrieval** → Frontend fetches pipelines, images, and analysis results
4. **Background Processing** → Celery tasks process satellite images and detect anomalies
5. **Alert Generation** → Analysis results trigger alerts automatically
6. **Dashboard Display** → Dashboard aggregates data from multiple sources

### Communication Flow Example

```
User Login → GET /api/auth/jwt/create/
    ↓
Token stored in localStorage
    ↓
GET /api/dashboard/stats/ (with token)
    ↓
GET /api/pipelines/ (with token)
    ↓
For each pipeline: GET /api/pipelines/{id}/recent-images/
    ↓
GET /api/satellite-images/{id}/analysis-results/
    ↓
GET /api/alerts/active/
    ↓
Frontend displays combined data
```

---

## Complete API Endpoints List

### 🔐 Authentication Endpoints (Djoser)

These endpoints are managed by the Djoser library and are accessible at `/api/auth/`:

| Method | Endpoint                                  | Description                      | Authentication Required |
| ------ | ----------------------------------------- | -------------------------------- | ----------------------- |
| POST   | `/api/auth/users/`                        | Register new user                | No                      |
| POST   | `/api/auth/jwt/create/`                   | Login and get JWT tokens         | No                      |
| POST   | `/api/auth/jwt/refresh/`                  | Refresh access token             | No                      |
| POST   | `/api/auth/jwt/logout/`                   | Logout (blacklist refresh token) | Yes                     |
| GET    | `/api/auth/users/me/`                     | Get current user info            | Yes                     |
| PUT    | `/api/auth/users/me/`                     | Update current user              | Yes                     |
| DELETE | `/api/auth/users/me/`                     | Delete current user              | Yes                     |
| POST   | `/api/auth/users/reset_password/`         | Request password reset           | No                      |
| POST   | `/api/auth/users/reset_password_confirm/` | Confirm password reset           | No                      |
| POST   | `/api/auth/users/set_password/`           | Change password                  | Yes                     |

---

### 👥 User Management Endpoints (`/api/users/`)

| Method | Endpoint                      | Description              | Auth | Permissions   |
| ------ | ----------------------------- | ------------------------ | ---- | ------------- |
| GET    | `/api/users/`                 | List all users           | Yes  | Admin only    |
| POST   | `/api/users/`                 | Create new user          | Yes  | Admin only    |
| GET    | `/api/users/{id}/`            | Get user details         | Yes  | Admin only    |
| PUT    | `/api/users/{id}/`            | Update user              | Yes  | Admin only    |
| PATCH  | `/api/users/{id}/`            | Partial update user      | Yes  | Admin only    |
| DELETE | `/api/users/{id}/`            | Delete user              | Yes  | Admin only    |
| GET    | `/api/users/me/`              | Get current user profile | Yes  | Authenticated |
| PUT    | `/api/users/me/`              | Update current user      | Yes  | Authenticated |
| POST   | `/api/users/{id}/activate/`   | Activate user            | Yes  | Admin only    |
| POST   | `/api/users/{id}/deactivate/` | Deactivate user          | Yes  | Admin only    |

**Supported Filters:**

- `search`: Search by email, first_name, last_name
- `is_staff`: Filter by staff status
- `is_superuser`: Filter by superuser status
- `is_active`: Filter by active status
- `ordering`: Order by date_joined, last_login, email

---

### 🏗️ Pipeline Endpoints (`/api/pipelines/`)

| Method | Endpoint                                | Description                   | Auth | Notes                |
| ------ | --------------------------------------- | ----------------------------- | ---- | -------------------- |
| GET    | `/api/pipelines/`                       | List all pipelines            | Yes  | Supports pagination  |
| POST   | `/api/pipelines/`                       | Create new pipeline           | Yes  | Requires coordinates |
| GET    | `/api/pipelines/{id}/`                  | Get pipeline details          | Yes  | -                    |
| PUT    | `/api/pipelines/{id}/`                  | Update pipeline               | Yes  | -                    |
| PATCH  | `/api/pipelines/{id}/`                  | Partial update pipeline       | Yes  | -                    |
| DELETE | `/api/pipelines/{id}/`                  | Delete pipeline               | Yes  | -                    |
| GET    | `/api/pipelines/{id}/segments/`         | Get all segments for pipeline | Yes  | Custom action        |
| GET    | `/api/pipelines/{id}/recent-images/`    | Get recent images (30 days)   | Yes  | Custom action        |
| GET    | `/api/pipelines/{id}/analysis-summary/` | Get analysis summary          | Yes  | Custom action        |

**Supported Filters:**

- `search`: Search by name, description, material
- `status`: Filter by active, maintenance, inactive
- `material`: Filter by steel, plastic, concrete, composite
- `ordering`: Order by name, length_km, created_at

**Query Parameters for Recent Images:**

- `days`: Number of days to look back (default: 30)

---

### 🛰️ Satellite Image Endpoints (`/api/satellite-images/`)

| Method | Endpoint                                       | Description                    | Auth | Notes               |
| ------ | ---------------------------------------------- | ------------------------------ | ---- | ------------------- |
| GET    | `/api/satellite-images/`                       | List all satellite images      | Yes  | Supports pagination |
| POST   | `/api/satellite-images/`                       | Create new satellite image     | Yes  | -                   |
| GET    | `/api/satellite-images/{id}/`                  | Get image details              | Yes  | -                   |
| PUT    | `/api/satellite-images/{id}/`                  | Update image                   | Yes  | -                   |
| PATCH  | `/api/satellite-images/{id}/`                  | Partial update image           | Yes  | -                   |
| DELETE | `/api/satellite-images/{id}/`                  | Delete image                   | Yes  | -                   |
| POST   | `/api/satellite-images/{id}/analyze/`          | Trigger image analysis         | Yes  | Custom action       |
| GET    | `/api/satellite-images/{id}/analysis-results/` | Get analysis results for image | Yes  | Custom action       |

**Supported Filters:**

- `search`: Search by satellite_name, sensor
- `pipeline`: Filter by pipeline ID
- `satellite_name`: Filter by satellite
- `processing_status`: Filter by pending, processing, completed, failed
- `source_api`: Filter by source API
- `ordering`: Order by image_date, created_at

---

### 🔍 Analysis Result Endpoints (`/api/analysis-results/`)

| Method | Endpoint                                          | Description                    | Auth | Notes               |
| ------ | ------------------------------------------------- | ------------------------------ | ---- | ------------------- |
| GET    | `/api/analysis-results/`                          | List all analysis results      | Yes  | Supports pagination |
| POST   | `/api/analysis-results/`                          | Create new analysis result     | Yes  | -                   |
| GET    | `/api/analysis-results/{id}/`                     | Get analysis result details    | Yes  | -                   |
| PUT    | `/api/analysis-results/{id}/`                     | Update analysis result         | Yes  | -                   |
| PATCH  | `/api/analysis-results/{id}/`                     | Partial update analysis result | Yes  | -                   |
| DELETE | `/api/analysis-results/{id}/`                     | Delete analysis result         | Yes  | -                   |
| POST   | `/api/analysis-results/{id}/verify/`              | Verify analysis result         | Yes  | Custom action       |
| POST   | `/api/analysis-results/{id}/mark-false-positive/` | Mark as false positive         | Yes  | Custom action       |
| POST   | `/api/analysis-results/{id}/create-alert/`        | Create alert from result       | Yes  | Custom action       |

**Supported Filters:**

- `search`: Search by analysis_type, description
- `analysis_type`: leak_detection, vegetation_encroachment, etc.
- `severity`: low, medium, high, critical
- `status`: pending, verified, false_positive, etc.
- `confidence_min`: Minimum confidence score
- `confidence_max`: Maximum confidence score
- `date_from`: Filter from date
- `date_to`: Filter to date
- `pipeline`: Filter by pipeline ID
- `verified`: Filter by verification status
- `ordering`: Order by confidence_score, created_at, severity

**Analysis Types:**

- `leak_detection`: Leak Detection
- `vegetation_encroachment`: Vegetation Encroachment
- `ground_subsidence`: Ground Subsidence
- `construction_activity`: Construction Activity
- `equipment_damage`: Equipment Damage
- `corrosion_detection`: Corrosion Detection
- `thermal_anomaly`: Thermal Anomaly
- `weather_damage`: Weather Damage

---

### 🚨 Monitoring Alert Endpoints (`/api/alerts/`)

| Method | Endpoint                    | Description                    | Auth | Notes               |
| ------ | --------------------------- | ------------------------------ | ---- | ------------------- |
| GET    | `/api/alerts/`              | List all alerts                | Yes  | Supports pagination |
| POST   | `/api/alerts/`              | Create new alert               | Yes  | -                   |
| GET    | `/api/alerts/{id}/`         | Get alert details              | Yes  | -                   |
| PUT    | `/api/alerts/{id}/`         | Update alert                   | Yes  | -                   |
| PATCH  | `/api/alerts/{id}/`         | Partial update alert           | Yes  | -                   |
| DELETE | `/api/alerts/{id}/`         | Delete alert                   | Yes  | -                   |
| POST   | `/api/alerts/{id}/resolve/` | Resolve alert                  | Yes  | Custom action       |
| GET    | `/api/alerts/active/`       | Get active (unresolved) alerts | Yes  | Custom action       |

**Supported Filters:**

- `search`: Search by message, alert_type
- `is_resolved`: Filter by resolved status (true/false)
- `priority`: low, medium, high, critical
- `alert_type`: Filter by alert type
- `ordering`: Order by priority, created_at

**Alert Types:**

- `leak_detected`
- `vegetation_encroachment`
- `ground_subsidence`
- `construction_activity`
- `equipment_damage`
- `corrosion_detected`
- `thermal_anomaly`
- `system_error`
- `maintenance_required`

---

### 📊 Dashboard Endpoints (`/api/dashboard/`)

| Method | Endpoint                          | Description                  | Auth | Notes         |
| ------ | --------------------------------- | ---------------------------- | ---- | ------------- |
| GET    | `/api/dashboard/stats/`           | Get dashboard statistics     | Yes  | Custom action |
| GET    | `/api/dashboard/recent-activity/` | Get recent activity          | Yes  | Custom action |
| GET    | `/api/dashboard/pipeline-health/` | Get pipeline health overview | Yes  | Custom action |

**Dashboard Stats Returns:**

- `total_pipelines`: Total number of pipelines
- `active_pipelines`: Number of active pipelines
- `total_images`: Total satellite images
- `pending_analyses`: Pending analysis results
- `active_alerts`: Unresolved alerts
- `critical_alerts`: Critical priority alerts
- `recent_anomalies`: Anomalies in last 7 days
- `last_analysis_date`: Last analysis timestamp

**Recent Activity Returns:**

- `recent_results`: Last 10 analysis results (7 days)
- `recent_alerts`: Last 10 alerts (7 days)
- `recent_images`: Last 10 satellite images (7 days)

**Pipeline Health Returns:**
Array of objects with:

- `pipeline_id`: UUID
- `pipeline_name`: Name
- `health_status`: critical, warning, healthy, unknown
- `total_analyses`: Number of analyses in last 30 days
- `critical_issues`: Number of critical issues
- `high_issues`: Number of high issues
- `last_analysis`: Timestamp of last analysis

---

### 🗺️ Pipeline Segment Endpoints (`/api/segments/`)

| Method | Endpoint                                | Description                | Auth | Notes               |
| ------ | --------------------------------------- | -------------------------- | ---- | ------------------- |
| GET    | `/api/segments/`                        | List all segments          | Yes  | Supports pagination |
| POST   | `/api/segments/`                        | Create new segment         | Yes  | -                   |
| GET    | `/api/segments/{id}/`                   | Get segment details        | Yes  | -                   |
| PUT    | `/api/segments/{id}/`                   | Update segment             | Yes  | -                   |
| PATCH  | `/api/segments/{id}/`                   | Partial update segment     | Yes  | -                   |
| DELETE | `/api/segments/{id}/`                   | Delete segment             | Yes  | -                   |
| POST   | `/api/segments/{id}/update-monitoring/` | Update last monitored time | Yes  | Custom action       |

**Supported Filters:**

- `search`: Search by segment_name
- `pipeline`: Filter by pipeline ID
- `risk_level`: low, medium, high, critical
- `terrain_type`: urban, rural, forest, water, mountain, desert
- `ordering`: Order by segment_number, risk_level, last_monitored

---

### ⚙️ Monitoring Configuration Endpoints (`/api/configurations/`)

| Method | Endpoint                    | Description                  | Auth | Notes               |
| ------ | --------------------------- | ---------------------------- | ---- | ------------------- |
| GET    | `/api/configurations/`      | List all configurations      | Yes  | Supports pagination |
| POST   | `/api/configurations/`      | Create new configuration     | Yes  | -                   |
| GET    | `/api/configurations/{id}/` | Get configuration details    | Yes  | -                   |
| PUT    | `/api/configurations/{id}/` | Update configuration         | Yes  | -                   |
| PATCH  | `/api/configurations/{id}/` | Partial update configuration | Yes  | -                   |
| DELETE | `/api/configurations/{id}/` | Delete configuration         | Yes  | -                   |

**Supported Filters:**

- `search`: Search by pipeline name

---

### 📝 Activity Log Endpoints (`/api/activity-logs/`)

| Method | Endpoint                   | Description              | Auth | Notes                 |
| ------ | -------------------------- | ------------------------ | ---- | --------------------- |
| GET    | `/api/activity-logs/`      | List all activity logs   | Yes  | Read-only, pagination |
| GET    | `/api/activity-logs/{id}/` | Get activity log details | Yes  | Read-only             |

**Supported Filters:**

- `search`: Search by action, resource_type, resource_name
- `action`: Filter by action type
- `resource_type`: Filter by resource type
- `user`: Filter by user ID
- `success`: Filter by success status (true/false)
- `ordering`: Order by created_at, action, user

**Action Types:**

- `create`, `update`, `delete`, `view`
- `login`, `logout`
- `analyze`, `alert`, `verify`, `resolve`

**Resource Types:**

- `pipeline`, `satellite_image`, `analysis_result`
- `alert`, `user`, `configuration`

---

### 🔧 System Settings Endpoints (`/api/settings/`)

| Method | Endpoint                         | Description                    | Auth | Permissions |
| ------ | -------------------------------- | ------------------------------ | ---- | ----------- |
| GET    | `/api/settings/`                 | List system settings           | Yes  | Admin only  |
| PUT    | `/api/settings/{id}/`            | Update system settings         | Yes  | Admin only  |
| PATCH  | `/api/settings/{id}/`            | Partial update system settings | Yes  | Admin only  |
| PUT    | `/api/settings/update-settings/` | Update settings (no ID needed) | Yes  | Admin only  |
| GET    | `/api/settings/current/`         | Get current system settings    | Yes  | Admin only  |

---

### 📖 API Documentation Endpoints

| Method | Endpoint        | Description              | Auth |
| ------ | --------------- | ------------------------ | ---- |
| GET    | `/swagger/`     | Swagger UI documentation | No   |
| GET    | `/redoc/`       | ReDoc documentation      | No   |
| GET    | `/swagger.json` | OpenAPI JSON schema      | No   |

---

## Data Flow Architecture

### 1. User Workflow

```
┌─────────────────┐
│   User Login    │
└────────┬────────┘
         │
         ↓
┌──────────────────────┐
│ POST /auth/jwt/     │
│ create/             │
│ Returns: tokens      │
└────────┬─────────────┘
         │
         ↓
┌──────────────────────┐
│ GET /users/me/      │
│ Get user profile     │
└────────┬─────────────┘
         │
         ↓
┌──────────────────────┐
│ GET /dashboard/     │
│ stats/              │
│ Returns: overview    │
└─────────────────────┘
```

### 2. Pipeline Monitoring Workflow

```
┌──────────────────────┐
│ GET /pipelines/      │
│ List pipelines       │
└────────┬─────────────┘
         │
         ↓
┌────────────────────────────┐
│ GET /pipelines/{id}/       │
│ Get pipeline details       │
└────────┬───────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ GET /pipelines/{id}/         │
│ recent-images/               │
│ Returns: recent images        │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ GET /satellite-images/{id}/  │
│ analysis-results/            │
│ Returns: analysis results     │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ GET /alerts/active/          │
│ Returns: active alerts        │
└──────────────────────────────┘
```

### 3. Analysis Workflow

```
┌──────────────────────────────┐
│ Background Task:             │
│ fetch_satellite_imagery_     │
│ periodic()                   │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ Create SatelliteImage record │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ Background Task:             │
│ analyze_satellite_image()    │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ Create AnalysisResult record │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ If critical: Create Alert    │
└──────────────────────────────┘
```

### 4. Data Relationships

```
Pipeline
  ├─ has many → SatelliteImage
  │              ├─ has many → AnalysisResult
  │              │               └─ can create → MonitoringAlert
  │              └─ has one → PipelineSegment
  │
  ├─ has many → PipelineSegment
  │
  └─ has one → MonitoringConfiguration

User
  ├─ can create → Pipeline
  ├─ can create → AnalysisResult (via verification)
  └─ receives → MonitoringAlert (via recipients field)

ActivityLog
  └─ records → User actions on resources

SystemSettings
  └─ system-wide configuration (singleton)
```

---

## Authentication Flow

### 1. Login Process

```typescript
// Frontend request
POST /api/auth/jwt/create/
{
  "email": "user@example.com",
  "password": "secret123"
}

// Response
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}

// Frontend stores tokens in localStorage
localStorage.setItem('access_token', access)
localStorage.setItem('refresh_token', refresh)
```

### 2. Making Authenticated Requests

```typescript
// All subsequent requests include Authorization header
GET /api/pipelines/
Headers: {
  "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

### 3. Token Refresh

```typescript
// When access token expires (401 error)
POST /api/auth/jwt/refresh/
{
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}

// Response
{
  "access": "new_access_token..."
}
```

### 4. Logout

```typescript
// Logout request
POST /
  api /
  auth /
  jwt /
  logout /
  {
    refresh: "refresh_token",
  };

// Frontend clears localStorage
localStorage.removeItem("access_token");
localStorage.removeItem("refresh_token");
```

---

## Background Tasks

The application uses Celery for background task processing. These tasks are triggered automatically or manually:

### Periodic Tasks (via Celery Beat)

| Task                               | Frequency     | Description                                  |
| ---------------------------------- | ------------- | -------------------------------------------- |
| `fetch_satellite_imagery_periodic` | Hourly        | Fetch satellite imagery for active pipelines |
| `analyze_images_periodic`          | Every 6 hours | Analyze pending satellite images             |
| `detect_anomalies_periodic`        | Daily         | Detect anomalies across pipelines            |
| `cleanup_old_data`                 | Weekly        | Clean up old data                            |
| `send_alert_notifications`         | Every 4 hours | Send notifications for unresolved alerts     |
| `generate_daily_report`            | Daily         | Generate daily monitoring report             |
| `update_pipeline_health_scores`    | Daily         | Update pipeline health scores                |

### Manual Tasks

| Task                                  | Trigger    | Description                            |
| ------------------------------------- | ---------- | -------------------------------------- |
| `fetch_pipeline_imagery(pipeline_id)` | Manual/API | Fetch imagery for specific pipeline    |
| `analyze_satellite_image(image_id)`   | Manual/API | Analyze specific image                 |
| `detect_anomalies(pipeline_id)`       | Manual     | Detect anomalies for specific pipeline |
| `send_alert_notification(alert_id)`   | Automatic  | Send specific alert notification       |
| `update_pipeline_health(pipeline_id)` | Manual     | Update health for specific pipeline    |
| `process_batch_images(image_ids)`     | Manual     | Process multiple images                |
| `export_monitoring_data(...)`         | Manual     | Export data for a pipeline             |

---

## Key Features

### 1. Geographic Data

- All geographic data uses PostGIS geometry fields
- Pipelines have LineString routes
- Images have Polygon bounds
- Analysis results have Point locations

### 2. Search & Filtering

- Full-text search on relevant fields
- Filter by any model field
- Ordering by any field (ascending/descending)
- Pagination for large result sets

### 3. Permissions

- **Public**: Authentication endpoints, API documentation
- **Authenticated**: Most CRUD operations
- **Admin Only**: User management, system settings

### 4. Data Validation

- Coordinates validated on create/update
- Confidence scores between 0-1
- File size limits enforced
- Required fields enforced by serializers

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful GET, PUT, PATCH
- `201 Created`: Successful POST
- `204 No Content`: Successful DELETE
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing/invalid token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource doesn't exist
- `500 Internal Server Error`: Server error

Error response format:

```json
{
  "detail": "Error message here"
}
```

Validation error format:

```json
{
  "field_name": ["Error message"]
}
```

---

## Summary

This API provides a comprehensive pipeline monitoring system with:

- **8 main resource types**: Pipelines, Satellite Images, Analysis Results, Alerts, Segments, Configurations, Activity Logs, System Settings
- **80+ endpoints** for CRUD operations and custom actions
- **JWT authentication** with token refresh
- **Background processing** via Celery tasks
- **Geographic data support** using PostGIS
- **Advanced filtering & search** capabilities
- **Role-based permissions** (public, authenticated, admin)
- **Comprehensive logging** of all user activities
