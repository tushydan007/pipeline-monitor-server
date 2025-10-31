from django.db import models
from django.contrib.auth import get_user_model
from django.contrib.gis.db import models as gis_models
from django.contrib.gis.geos import Point, LineString, Polygon
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
import logging

logger = logging.getLogger(__name__)

# Import custom validators for file uploads
from .validators import validate_tiff_file, validate_thumbnail_file

# Get the custom User model
User = get_user_model()


class Pipeline(models.Model):
    """Model representing a pipeline infrastructure"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    length_km = models.FloatField(validators=[MinValueValidator(0.1)])
    diameter_mm = models.FloatField(validators=[MinValueValidator(10)])
    material = models.CharField(
        max_length=100,
        choices=[
            ("steel", "Steel"),
            ("plastic", "Plastic"),
            ("concrete", "Concrete"),
            ("composite", "Composite"),
        ],
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("active", "Active"),
            ("maintenance", "Maintenance"),
            ("inactive", "Inactive"),
        ],
        default="active",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Geographic information
    start_point = gis_models.PointField(null=True, blank=True)
    end_point = gis_models.PointField(null=True, blank=True)
    route = gis_models.LineStringField(
        null=True, blank=True, help_text="Pipeline route geometry"
    )
    default_bounds = gis_models.PolygonField(
        null=True,
        blank=True,
        help_text="Automatically calculated satellite image bounds from route",
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.length_km}km)"

    def calculate_default_bounds(self, buffer_km=1.0):
        """
        Calculate default satellite image bounds from the pipeline route.

        Args:
            buffer_km: Buffer distance in kilometers to add around the route (default: 1km)

        Returns:
            Polygon representing the bounds, or None if route is not available
        """
        if not self.route:
            return None

        try:
            # Convert buffer from km to degrees (approximate: 1 degree ≈ 111 km)
            buffer_degrees = buffer_km / 111.0

            # Get the extent of the route
            extent = (
                self.route.extent
            )  # (minx, miny, maxx, maxy) => (west, south, east, north)

            # Add buffer
            min_lon = extent[0] - buffer_degrees
            min_lat = extent[1] - buffer_degrees
            max_lon = extent[2] + buffer_degrees
            max_lat = extent[3] + buffer_degrees

            # Create polygon bounds
            bounds = Polygon.from_bbox((min_lon, min_lat, max_lon, max_lat))
            bounds.srid = self.route.srid

            return bounds
        except Exception as e:
            logger.error(f"Error calculating bounds for pipeline {self.id}: {str(e)}")
            return None


class SatelliteImage(models.Model):
    """Model for storing satellite imagery data"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        Pipeline, on_delete=models.CASCADE, related_name="satellite_images"
    )
    # User associations
    # The end user whose dashboard should display results
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="user_satellite_images",
        null=True,
        blank=True,
    )
    # The admin/staff who uploaded the image
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        related_name="uploaded_satellite_images",
        null=True,
        blank=True,
    )

    # Image metadata
    image_date = models.DateTimeField()
    satellite_name = models.CharField(max_length=100)
    sensor = models.CharField(max_length=50)
    resolution_m = models.FloatField(help_text="Spatial resolution in meters")

    # Geographic bounds
    bounds = gis_models.PolygonField(
        null=True, blank=True, help_text="Image coverage area"
    )
    center_point = gis_models.PointField(null=True, blank=True)

    # Image files - TIFF format only
    image_file = models.FileField(
        upload_to="satellite_images/",
        validators=[validate_tiff_file],
        help_text="Satellite image file in TIFF format (.tif, .tiff). Maximum size: 100 MB",
    )
    thumbnail = models.ImageField(
        upload_to="thumbnails/",
        blank=True,
        null=True,
        validators=[validate_thumbnail_file],
        help_text="Thumbnail image (JPG, PNG, or TIFF). Maximum size: 5 MB",
    )

    # Processing status
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )

    # Source information
    # For uploaded images, this will be 'uploaded'.
    source_api = models.CharField(max_length=50, default="uploaded")
    api_image_id = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-image_date"]
        unique_together = ["pipeline", "image_date", "satellite_name"]

    def __str__(self):
        return f"{self.satellite_name} - {self.image_date.strftime('%Y-%m-%d')}"


class AnalysisResult(models.Model):
    """Model for storing analysis results from satellite imagery"""

    # Choices constants
    ANALYSIS_TYPE_CHOICES = [
        ("leak_detection", "Leak Detection"),
        ("vegetation_encroachment", "Vegetation Encroachment"),
        ("ground_subsidence", "Ground Subsidence"),
        ("construction_activity", "Construction Activity"),
        ("equipment_damage", "Equipment Damage"),
        ("corrosion_detection", "Corrosion Detection"),
        ("thermal_anomaly", "Thermal Anomaly"),
        ("weather_damage", "Weather Damage"),
    ]

    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    STATUS_CHOICES = [
        ("pending", "Pending Review"),
        ("verified", "Verified"),
        ("false_positive", "False Positive"),
        ("alert_created", "Alert Created"),
        ("resolved", "Resolved"),
        ("dismissed", "Dismissed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    satellite_image = models.ForeignKey(
        SatelliteImage, on_delete=models.CASCADE, related_name="analysis_results"
    )

    # Analysis metadata
    analysis_type = models.CharField(max_length=50, choices=ANALYSIS_TYPE_CHOICES)

    # Results
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Confidence score between 0 and 1",
    )
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)

    # Geographic information
    detected_location = gis_models.PointField(
        null=True, blank=True, help_text="Location of detected anomaly"
    )
    affected_area = gis_models.PolygonField(
        blank=True, null=True, help_text="Area affected by the anomaly"
    )

    # Analysis details
    description = models.TextField(blank=True)
    raw_data = models.JSONField(blank=True, null=True, help_text="Raw analysis data")

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Verification
    verified_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )
    verified_at = models.DateTimeField(null=True, blank=True)
    verification_notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.analysis_type} - {self.confidence_score:.2f} confidence"


class MonitoringAlert(models.Model):
    """Model for storing monitoring alerts and notifications"""

    # Choices constants
    ALERT_TYPE_CHOICES = [
        ("leak_detected", "Leak Detected"),
        ("vegetation_encroachment", "Vegetation Encroachment"),
        ("ground_subsidence", "Ground Subsidence"),
        ("construction_activity", "Construction Activity"),
        ("equipment_damage", "Equipment Damage"),
        ("corrosion_detected", "Corrosion Detected"),
        ("thermal_anomaly", "Thermal Anomaly"),
        ("system_error", "System Error"),
        ("maintenance_required", "Maintenance Required"),
    ]

    PRIORITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis_result = models.ForeignKey(
        AnalysisResult, on_delete=models.CASCADE, related_name="alerts"
    )

    # Alert details
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPE_CHOICES)

    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES)

    message = models.TextField()
    is_resolved = models.BooleanField(default=False)

    # Notification
    sent_at = models.DateTimeField(null=True, blank=True)
    recipients = models.ManyToManyField(User, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.alert_type} - {self.priority} priority"


class PipelineSegment(models.Model):
    """Model for dividing pipeline into monitoring segments"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        Pipeline, on_delete=models.CASCADE, related_name="segments"
    )

    # Segment information
    segment_name = models.CharField(max_length=100)
    segment_number = models.PositiveIntegerField()
    length_km = models.FloatField(validators=[MinValueValidator(0.1)])

    # Geographic information
    start_point = gis_models.PointField(null=True, blank=True)
    end_point = gis_models.PointField(null=True, blank=True)
    geometry = gis_models.LineStringField(null=True, blank=True)

    # Monitoring parameters
    monitoring_frequency_hours = models.PositiveIntegerField(default=24)
    last_monitored = models.DateTimeField(null=True, blank=True)

    # Risk assessment
    risk_level = models.CharField(
        max_length=20,
        choices=[
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
            ("critical", "Critical"),
        ],
        default="medium",
    )

    # Environmental factors
    terrain_type = models.CharField(
        max_length=50,
        choices=[
            ("urban", "Urban"),
            ("rural", "Rural"),
            ("forest", "Forest"),
            ("water", "Water Body"),
            ("mountain", "Mountain"),
            ("desert", "Desert"),
        ],
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["pipeline", "segment_number"]
        unique_together = ["pipeline", "segment_number"]

    def __str__(self):
        return f"{self.pipeline.name} - Segment {self.segment_number}"


class MonitoringConfiguration(models.Model):
    """Model for storing monitoring system configuration"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.OneToOneField(
        Pipeline, on_delete=models.CASCADE, related_name="monitoring_config"
    )

    # API Configuration
    nasa_api_key = models.CharField(max_length=200, blank=True)
    nasa_api_enabled = models.BooleanField(default=True)

    # Analysis Configuration
    analysis_enabled = models.BooleanField(default=True)
    auto_analysis = models.BooleanField(default=True)
    confidence_threshold = models.FloatField(
        default=0.7, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    # Alert Configuration
    alerts_enabled = models.BooleanField(default=True)
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)

    # Monitoring Schedule
    monitoring_frequency_hours = models.PositiveIntegerField(default=24)
    analysis_frequency_hours = models.PositiveIntegerField(default=6)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Config for {self.pipeline.name}"


class ActivityLog(models.Model):
    """Model for storing user activity logs"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    # Activity details
    action = models.CharField(
        max_length=50,
        choices=[
            ("create", "Create"),
            ("update", "Update"),
            ("delete", "Delete"),
            ("view", "View"),
            ("login", "Login"),
            ("logout", "Logout"),
            ("analyze", "Analyze"),
            ("alert", "Alert"),
            ("verify", "Verify"),
            ("resolve", "Resolve"),
        ],
    )

    # Resource information
    resource_type = models.CharField(
        max_length=50,
        choices=[
            ("pipeline", "Pipeline"),
            ("satellite_image", "Satellite Image"),
            ("analysis_result", "Analysis Result"),
            ("alert", "Alert"),
            ("user", "User"),
            ("configuration", "Configuration"),
        ],
    )
    resource_id = models.UUIDField()
    resource_name = models.CharField(max_length=200)

    # Additional details
    details = models.JSONField(blank=True, null=True)

    # Request information
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)

    # Status
    success = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["user"]),
            models.Index(fields=["action"]),
            models.Index(fields=["resource_type"]),
        ]

    def __str__(self):
        return f"{self.action} on {self.resource_type} by {self.user}"


class SystemSettings(models.Model):
    """Model for storing system-wide settings"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # General Settings
    site_name = models.CharField(max_length=200, default="Pipeline Monitoring System")
    site_description = models.TextField(
        default="Advanced pipeline monitoring and analysis platform"
    )
    timezone = models.CharField(max_length=50, default="UTC")
    language = models.CharField(max_length=10, default="en")
    theme = models.CharField(
        max_length=10,
        choices=[
            ("light", "Light"),
            ("dark", "Dark"),
            ("auto", "Auto"),
        ],
        default="light",
    )

    # Notification Settings
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    push_notifications = models.BooleanField(default=True)
    alert_email = models.EmailField(default="admin@example.com")
    alert_phone = models.CharField(max_length=20, default="+1234567890")

    # Analysis Settings
    analysis_confidence_threshold = models.FloatField(
        default=0.8, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    auto_analysis_enabled = models.BooleanField(default=True)
    analysis_retention_days = models.PositiveIntegerField(default=365)

    # Security Settings
    session_timeout = models.PositiveIntegerField(default=30)  # minutes
    password_min_length = models.PositiveIntegerField(default=8)
    require_2fa = models.BooleanField(default=False)
    failed_login_lockout = models.PositiveIntegerField(default=5)

    # API Settings
    api_rate_limit = models.PositiveIntegerField(default=1000)  # per hour
    api_key_expiry_days = models.PositiveIntegerField(default=90)

    # Storage Settings
    max_file_size_mb = models.PositiveIntegerField(default=100)
    storage_quota_gb = models.PositiveIntegerField(default=1000)
    auto_cleanup_enabled = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Settings"
        verbose_name_plural = "System Settings"

    def __str__(self):
        return "System Settings"
