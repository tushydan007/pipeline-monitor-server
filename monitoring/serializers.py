from rest_framework import serializers
from django.contrib.gis.geos import Point, LineString, Polygon
from django.contrib.auth import get_user_model
from user.serializers import UserSerializer
import json
import logging

# Use custom User model
User = get_user_model()
from .models import (
    Pipeline,
    SatelliteImage,
    AnalysisResult,
    MonitoringAlert,
    PipelineSegment,
    MonitoringConfiguration,
    ActivityLog,
    SystemSettings,
)

logger = logging.getLogger(__name__)


class PipelineSerializer(serializers.ModelSerializer):
    """Serializer for Pipeline model"""

    start_lat = serializers.FloatField(write_only=True)
    start_lon = serializers.FloatField(write_only=True)
    end_lat = serializers.FloatField(write_only=True)
    end_lon = serializers.FloatField(write_only=True)

    start_point_lat = serializers.SerializerMethodField()
    start_point_lon = serializers.SerializerMethodField()
    end_point_lat = serializers.SerializerMethodField()
    end_point_lon = serializers.SerializerMethodField()
    route_coordinates = serializers.SerializerMethodField()
    default_bounds_overlay = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Pipeline
        fields = [
            "id",
            "name",
            "description",
            "length_km",
            "diameter_mm",
            "material",
            "status",
            "created_at",
            "updated_at",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "start_point_lat",
            "start_point_lon",
            "end_point_lat",
            "end_point_lon",
            "route_coordinates",
            "default_bounds",
            "default_bounds_overlay",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_start_point_lat(self, obj):
        return obj.start_point.y if obj.start_point else None

    def get_start_point_lon(self, obj):
        return obj.start_point.x if obj.start_point else None

    def get_end_point_lat(self, obj):
        return obj.end_point.y if obj.end_point else None

    def get_end_point_lon(self, obj):
        return obj.end_point.x if obj.end_point else None

    def get_route_coordinates(self, obj):
        """Return route coordinates as array of [lat, lon] pairs for frontend display"""
        if obj.route:
            # Extract all coordinates from the LineString
            # LineString.coords returns tuples of (x, y) = (longitude, latitude)
            # We need to convert to [[lat, lon], ...] format for Leaflet (which expects [lat, lon])
            try:
                # Method 1: Try using coords property directly
                # Convert to list first to ensure we get all coordinates
                if hasattr(obj.route, "coords"):
                    coords_list = []
                    # Iterate through all coordinate tuples
                    coords_iter = obj.route.coords
                    # Convert generator/iterator to list to get all points
                    coords_tuples = (
                        list(coords_iter)
                        if not isinstance(coords_iter, list)
                        else coords_iter
                    )

                    for coord in coords_tuples:
                        # Handle both tuple and list formats
                        if isinstance(coord, (tuple, list)) and len(coord) >= 2:
                            # coord is (longitude, latitude) = (x, y)
                            lon = float(coord[0])
                            lat = float(coord[1])
                            # Convert to [latitude, longitude] for Leaflet
                            coords_list.append([lat, lon])

                    # Ensure we have at least 2 points
                    if len(coords_list) >= 2:
                        return coords_list

                # Method 2: Alternative approach using GeoJSON representation
                # This ensures we get all coordinates even if coords property has issues
                if hasattr(obj.route, "json"):
                    geojson_data = json.loads(obj.route.json)
                    if geojson_data.get("type") == "LineString":
                        coordinates = geojson_data.get("coordinates", [])
                        # GeoJSON coordinates are [longitude, latitude]
                        # Convert to [latitude, longitude] for Leaflet
                        coords_list = [
                            [coord[1], coord[0]]
                            for coord in coordinates
                            if len(coord) >= 2
                        ]
                        if len(coords_list) >= 2:
                            return coords_list

            except (AttributeError, IndexError, TypeError, ValueError) as e:
                # If there's any error extracting coordinates, log and fall back
                logger.warning(f"Error extracting route coordinates: {str(e)}")
                pass

        # Fallback to start and end points if no route or if extraction failed
        if obj.start_point and obj.end_point:
            return [
                [obj.start_point.y, obj.start_point.x],
                [obj.end_point.y, obj.end_point.x],
            ]
        return None

    def get_default_bounds_overlay(self, obj):
        """Return default_bounds in [[south, west], [north, east]] format for Leaflet"""
        if obj.default_bounds:
            extent = obj.default_bounds.extent  # (minx, miny, maxx, maxy) => (west, south, east, north)
            return [[extent[1], extent[0]], [extent[3], extent[2]]]
        return None

    def create(self, validated_data):
        # Extract lat/lon data
        start_lat = validated_data.pop("start_lat")
        start_lon = validated_data.pop("start_lon")
        end_lat = validated_data.pop("end_lat")
        end_lon = validated_data.pop("end_lon")

        # Create Point objects
        start_point = Point(start_lon, start_lat)
        end_point = Point(end_lon, end_lat)

        # Create LineString for route
        route = LineString([start_point, end_point])

        # Calculate default bounds from route
        temp_pipeline = Pipeline(route=route)
        default_bounds = temp_pipeline.calculate_default_bounds(buffer_km=1.0)

        # Create pipeline
        pipeline = Pipeline.objects.create(
            start_point=start_point, 
            end_point=end_point, 
            route=route, 
            default_bounds=default_bounds,
            **validated_data
        )

        return pipeline

    def update(self, instance, validated_data):
        # Handle coordinate updates
        if "start_lat" in validated_data and "start_lon" in validated_data:
            start_lat = validated_data.pop("start_lat")
            start_lon = validated_data.pop("start_lon")
            instance.start_point = Point(start_lon, start_lat)

        if "end_lat" in validated_data and "end_lon" in validated_data:
            end_lat = validated_data.pop("end_lat")
            end_lon = validated_data.pop("end_lon")
            instance.end_point = Point(end_lon, end_lat)

        # Update route if coordinates changed
        route_updated = False
        if instance.start_point and instance.end_point:
            new_route = LineString([instance.start_point, instance.end_point])
            if instance.route != new_route:
                instance.route = new_route
                route_updated = True

        # Update other fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        instance.save()
        
        # Recalculate default bounds if route was updated
        if route_updated and instance.route:
            default_bounds = instance.calculate_default_bounds(buffer_km=1.0)
            if default_bounds:
                instance.default_bounds = default_bounds
                instance.save(update_fields=['default_bounds'])
        
        return instance


class SatelliteImageSerializer(serializers.ModelSerializer):
    """Serializer for SatelliteImage model"""

    bounds_lat_min = serializers.FloatField(write_only=True, required=False)
    bounds_lon_min = serializers.FloatField(write_only=True, required=False)
    bounds_lat_max = serializers.FloatField(write_only=True, required=False)
    bounds_lon_max = serializers.FloatField(write_only=True, required=False)

    center_lat = serializers.SerializerMethodField()
    center_lon = serializers.SerializerMethodField()
    pipeline_id = serializers.SerializerMethodField()
    # New fields
    user_id = serializers.UUIDField(source="user.id", read_only=True)
    uploaded_by_id = serializers.UUIDField(source="uploaded_by.id", read_only=True)
    # Read-only overlay bounds for front-end overlays: [[south, west], [north, east]]
    overlay_bounds = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = SatelliteImage
        fields = [
            "id",
            "pipeline",
            "pipeline_id",
            "user",
            "user_id",
            "uploaded_by_id",
            "image_date",
            "satellite_name",
            "sensor",
            "resolution_m",
            "image_file",
            "thumbnail",
            "processing_status",
            "source_api",
            "api_image_id",
            "created_at",
            "updated_at",
            "bounds_lat_min",
            "bounds_lon_min",
            "bounds_lat_max",
            "bounds_lon_max",
            "center_lat",
            "center_lon",
            "overlay_bounds",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_center_lat(self, obj):
        return obj.center_point.y if obj.center_point else None

    def get_center_lon(self, obj):
        return obj.center_point.x if obj.center_point else None

    def get_pipeline_id(self, obj):
        return str(obj.pipeline.id) if obj.pipeline else None

    def get_overlay_bounds(self, obj):
        # Return bbox in [[south, west], [north, east]] if bounds exist
        if obj.bounds:
            extent = (
                obj.bounds.extent
            )  # (minx, miny, maxx, maxy) => (west, south, east, north)
            return [[extent[1], extent[0]], [extent[3], extent[2]]]
        # Fallback: approximate from center if available
        if obj.center_point:
            lat = obj.center_point.y
            lon = obj.center_point.x
            dim = 0.1
            return [[lat - dim / 2, lon - dim / 2], [lat + dim / 2, lon + dim / 2]]
        return None

    def create(self, validated_data):
        # Ensure source_api defaults to 'uploaded'
        if "source_api" not in validated_data or not validated_data.get("source_api"):
            validated_data["source_api"] = "uploaded"

        # Handle bounds creation
        bounds_data = {}
        if all(
            key in validated_data
            for key in [
                "bounds_lat_min",
                "bounds_lon_min",
                "bounds_lat_max",
                "bounds_lon_max",
            ]
        ):
            lat_min = validated_data.pop("bounds_lat_min")
            lon_min = validated_data.pop("bounds_lon_min")
            lat_max = validated_data.pop("bounds_lat_max")
            lon_max = validated_data.pop("bounds_lon_max")

            # Create polygon bounds
            bounds = Polygon.from_bbox((lon_min, lat_min, lon_max, lat_max))
            bounds_data["bounds"] = bounds

            # Calculate center point
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            bounds_data["center_point"] = Point(center_lon, center_lat)

        # Create satellite image
        satellite_image = SatelliteImage.objects.create(**bounds_data, **validated_data)

        return satellite_image


class AnalysisResultSerializer(serializers.ModelSerializer):
    """Serializer for AnalysisResult model"""

    detected_lat = serializers.FloatField(write_only=True, required=False)
    detected_lon = serializers.FloatField(write_only=True, required=False)

    detected_location_lat = serializers.SerializerMethodField()
    detected_location_lon = serializers.SerializerMethodField()
    confidence_score_percentage = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisResult
        fields = [
            "id",
            "satellite_image",
            "analysis_type",
            "confidence_score",
            "confidence_score_percentage",
            "severity",
            "description",
            "raw_data",
            "status",
            "verified_by",
            "verified_at",
            "verification_notes",
            "created_at",
            "updated_at",
            "detected_lat",
            "detected_lon",
            "detected_location_lat",
            "detected_location_lon",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_detected_location_lat(self, obj):
        return obj.detected_location.y if obj.detected_location else None

    def get_detected_location_lon(self, obj):
        return obj.detected_location.x if obj.detected_location else None

    def get_confidence_score_percentage(self, obj):
        """Convert confidence score from 0-1 to 0-100 for frontend"""
        return int(obj.confidence_score * 100)

    def to_representation(self, instance):
        """Convert representation to use percentage"""
        data = super().to_representation(instance)
        # Override confidence_score with percentage for frontend
        data["confidence_score"] = data.pop("confidence_score_percentage")
        return data

    def create(self, validated_data):
        # Handle detected location
        if "detected_lat" in validated_data and "detected_lon" in validated_data:
            lat = validated_data.pop("detected_lat")
            lon = validated_data.pop("detected_lon")
            validated_data["detected_location"] = Point(lon, lat)

        return AnalysisResult.objects.create(**validated_data)


class MonitoringAlertSerializer(serializers.ModelSerializer):
    """Serializer for MonitoringAlert model"""

    analysis_result = AnalysisResultSerializer(read_only=True)
    analysis_result_id = serializers.UUIDField(write_only=True)

    class Meta:
        model = MonitoringAlert
        fields = [
            "id",
            "analysis_result",
            "analysis_result_id",
            "alert_type",
            "priority",
            "message",
            "is_resolved",
            "sent_at",
            "recipients",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class PipelineSegmentSerializer(serializers.ModelSerializer):
    """Serializer for PipelineSegment model"""

    start_lat = serializers.FloatField(write_only=True)
    start_lon = serializers.FloatField(write_only=True)
    end_lat = serializers.FloatField(write_only=True)
    end_lon = serializers.FloatField(write_only=True)

    start_point_lat = serializers.SerializerMethodField()
    start_point_lon = serializers.SerializerMethodField()
    end_point_lat = serializers.SerializerMethodField()
    end_point_lon = serializers.SerializerMethodField()

    class Meta:
        model = PipelineSegment
        fields = [
            "id",
            "pipeline",
            "segment_name",
            "segment_number",
            "length_km",
            "monitoring_frequency_hours",
            "last_monitored",
            "risk_level",
            "terrain_type",
            "created_at",
            "updated_at",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "start_point_lat",
            "start_point_lon",
            "end_point_lat",
            "end_point_lon",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_start_point_lat(self, obj):
        return obj.start_point.y if obj.start_point else None

    def get_start_point_lon(self, obj):
        return obj.start_point.x if obj.start_point else None

    def get_end_point_lat(self, obj):
        return obj.end_point.y if obj.end_point else None

    def get_end_point_lon(self, obj):
        return obj.end_point.x if obj.end_point else None

    def create(self, validated_data):
        # Extract lat/lon data
        start_lat = validated_data.pop("start_lat")
        start_lon = validated_data.pop("start_lon")
        end_lat = validated_data.pop("end_lat")
        end_lon = validated_data.pop("end_lon")

        # Create Point objects
        start_point = Point(start_lon, start_lat)
        end_point = Point(end_lon, end_lat)

        # Create LineString for geometry
        geometry = LineString([start_point, end_point])

        # Create segment
        segment = PipelineSegment.objects.create(
            start_point=start_point,
            end_point=end_point,
            geometry=geometry,
            **validated_data,
        )

        return segment


class MonitoringConfigurationSerializer(serializers.ModelSerializer):
    """Serializer for MonitoringConfiguration model"""

    pipeline_name = serializers.CharField(source="pipeline.name", read_only=True)

    class Meta:
        model = MonitoringConfiguration
        fields = [
            "id",
            "pipeline",
            "pipeline_name",
            "nasa_api_key",
            "nasa_api_enabled",
            "analysis_enabled",
            "auto_analysis",
            "confidence_threshold",
            "alerts_enabled",
            "email_notifications",
            "sms_notifications",
            "monitoring_frequency_hours",
            "analysis_frequency_hours",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class DashboardStatsSerializer(serializers.Serializer):
    """Serializer for dashboard statistics"""

    total_pipelines = serializers.IntegerField()
    active_pipelines = serializers.IntegerField()
    total_images = serializers.IntegerField()
    pending_analyses = serializers.IntegerField()
    active_alerts = serializers.IntegerField()
    critical_alerts = serializers.IntegerField()
    recent_anomalies = serializers.IntegerField()
    last_analysis_date = serializers.DateTimeField()


class ActivityLogSerializer(serializers.ModelSerializer):
    """Serializer for ActivityLog model"""

    user = UserSerializer(read_only=True)
    timestamp = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = ActivityLog
        fields = [
            "id",
            "user",
            "action",
            "resource_type",
            "resource_id",
            "resource_name",
            "details",
            "ip_address",
            "user_agent",
            "success",
            "timestamp",
            "created_at",
        ]
        read_only_fields = ["id", "created_at", "timestamp"]


class SystemSettingsSerializer(serializers.ModelSerializer):
    """Serializer for SystemSettings model"""

    class Meta:
        model = SystemSettings
        fields = [
            "id",
            "site_name",
            "site_description",
            "timezone",
            "language",
            "theme",
            "email_notifications",
            "sms_notifications",
            "push_notifications",
            "alert_email",
            "alert_phone",
            "analysis_confidence_threshold",
            "auto_analysis_enabled",
            "analysis_retention_days",
            "session_timeout",
            "password_min_length",
            "require_2fa",
            "failed_login_lockout",
            "api_rate_limit",
            "api_key_expiry_days",
            "max_file_size_mb",
            "storage_quota_gb",
            "auto_cleanup_enabled",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]
