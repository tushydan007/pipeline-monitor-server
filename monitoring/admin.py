from django.contrib import admin
from django.contrib.gis.admin import GISModelAdmin
from django.utils.html import format_html
from django.contrib import messages
from django.contrib.gis.geos import Point, LineString
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
from .forms import PipelineGeoJSONUploadForm


@admin.register(Pipeline)
class PipelineAdmin(GISModelAdmin):
    form = PipelineGeoJSONUploadForm
    list_display = ["name", "length_km", "material", "status", "created_at"]
    list_filter = ["status", "material", "created_at"]
    search_fields = ["name", "description"]
    readonly_fields = ["id", "created_at", "updated_at"]

    # Map configuration - Nigeria center
    default_lat = 9.0820
    default_lon = 8.6753
    default_zoom = 7
    map_width = 900
    map_height = 600

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "id",
                    "name",
                    "description",
                    "length_km",
                    "diameter_mm",
                    "material",
                    "status",
                )
            },
        ),
        (
            "GeoJSON Upload (Alternative to Manual Entry)",
            {
                "fields": ("geojson_file",),
                "description": "Upload a GeoJSON file containing a LineString geometry. This will automatically populate the geographic fields below. You can also manually set points on the map instead.",
                "classes": ("collapse",),
            },
        ),
        (
            "Geographic Information",
            {
                "fields": ("start_point", "end_point", "route"),
                "description": "These fields are optional. You can either upload a GeoJSON file above OR manually set points on the map below.",
                "classes": ("wide",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    class Media:
        css = {"all": ("admin/css/pipeline_admin.css",)}
        js = ("admin/js/pipeline_admin.js",)


@admin.register(SatelliteImage)
class SatelliteImageAdmin(GISModelAdmin):
    list_display = [
        "satellite_name",
        "image_date",
        "pipeline",
        "user",
        "uploaded_by",
        "processing_status",
        "resolution_m",
    ]
    list_filter = [
        "satellite_name",
        "processing_status",
        "source_api",
        "image_date",
        "user",
    ]
    search_fields = ["satellite_name", "sensor", "api_image_id", "user__email"]
    readonly_fields = ["id", "created_at", "updated_at", "image_preview"]

    # Map configuration - Nigeria center
    default_lat = 9.0820
    default_lon = 8.6753
    default_zoom = 7
    map_width = 900
    map_height = 600

    fieldsets = (
        (
            "Image Information",
            {
                "fields": (
                    "id",
                    "pipeline",
                    "user",
                    "uploaded_by",
                    "image_date",
                    "satellite_name",
                    "sensor",
                    "resolution_m",
                )
            },
        ),
        ("Files", {"fields": ("image_file", "image_preview", "thumbnail")}),
        ("Geographic Information", {"fields": ("bounds", "center_point")}),
        ("Processing", {"fields": ("processing_status", "source_api", "api_image_id")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def image_preview(self, obj):
        if obj.image_file:
            return format_html(
                '<img src="{}" width="200" height="150" style="border-radius: 5px;" />',
                obj.image_file.url,
            )
        return "No image"

    image_preview.short_description = "Preview"

    actions = ["derive_pipeline_route_from_image"]

    def derive_pipeline_route_from_image(self, request, queryset):
        """Admin action to derive a pipeline route (start, mid, end) from a TIFF's geo-bounds/center.

        Heuristic:
        - start: SW corner of image bounds
        - mid: image center (center_point)
        - end: NE corner of image bounds
        Updates the related Pipeline's start_point, end_point and route.
        """
        updated = 0
        skipped = 0
        for img in queryset:
            try:
                pipeline = getattr(img, "pipeline", None)
                if pipeline is None:
                    skipped += 1
                    continue
                # Require at least bounds or center to exist
                if not img.bounds and not img.center_point:
                    skipped += 1
                    continue

                # Derive corners from bounds when available
                start_pt = None
                end_pt = None
                mid_pt = None

                if img.bounds:
                    # bounds.extent: (minx, miny, maxx, maxy) => (west, south, east, north)
                    west, south, east, north = img.bounds.extent
                    start_pt = Point(west, south)
                    end_pt = Point(east, north)

                if img.center_point:
                    mid_pt = Point(img.center_point.x, img.center_point.y)

                # Fallbacks if some pieces missing
                if mid_pt is None and img.bounds:
                    mid_pt = Point((west + east) / 2.0, (south + north) / 2.0)
                if start_pt is None and img.center_point:
                    # Approximate ~0.01 degrees SW from center
                    start_pt = Point(
                        img.center_point.x - 0.01, img.center_point.y - 0.01
                    )
                if end_pt is None and img.center_point:
                    # Approximate ~0.01 degrees NE from center
                    end_pt = Point(img.center_point.x + 0.01, img.center_point.y + 0.01)

                if not (start_pt and end_pt):
                    skipped += 1
                    continue

                # Update pipeline geometry
                pipeline.start_point = start_pt
                pipeline.end_point = end_pt

                if mid_pt:
                    pipeline.route = LineString([start_pt, mid_pt, end_pt])
                else:
                    pipeline.route = LineString([start_pt, end_pt])

                pipeline.save()
                updated += 1
            except Exception:
                skipped += 1
                continue

        if updated:
            self.message_user(
                request,
                f"Updated pipeline geometry for {updated} image(s).",
                level=messages.SUCCESS,
            )
        if skipped:
            self.message_user(
                request,
                f"Skipped {skipped} image(s) due to missing bounds/center or pipeline.",
                level=messages.WARNING,
            )

    derive_pipeline_route_from_image.short_description = (
        "Derive pipeline route from selected image(s)"
    )


@admin.register(AnalysisResult)
class AnalysisResultAdmin(GISModelAdmin):
    list_display = ["confidence_score", "severity", "status", "created_at"]
    list_filter = ["severity", "status", "created_at"]
    search_fields = ["description"]
    readonly_fields = ["id", "created_at", "updated_at"]

    # Map configuration - Nigeria center
    default_lat = 9.0820
    default_lon = 8.6753
    default_zoom = 7
    map_width = 900
    map_height = 600

    fieldsets = (
        (
            "Analysis Information",
            {
                "fields": (
                    "id",
                    "satellite_image",
                    "confidence_score",
                    "severity",
                )
            },
        ),
        ("Location", {"fields": ("detected_location", "affected_area")}),
        ("Details", {"fields": ("description", "raw_data")}),
        (
            "Status",
            {"fields": ("status", "verified_by", "verified_at", "verification_notes")},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MonitoringAlert)
class MonitoringAlertAdmin(admin.ModelAdmin):
    list_display = ["priority", "is_resolved", "created_at"]
    list_filter = ["priority", "is_resolved", "created_at"]
    search_fields = ["message"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            "Alert Information",
            {"fields": ("id", "analysis_result", "alert_type", "priority", "message")},
        ),
        ("Status", {"fields": ("is_resolved", "sent_at", "recipients")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(PipelineSegment)
class PipelineSegmentAdmin(GISModelAdmin):
    list_display = [
        "segment_name",
        "pipeline",
        "segment_number",
        "risk_level",
        "terrain_type",
    ]
    list_filter = ["risk_level", "terrain_type", "pipeline"]
    search_fields = ["segment_name", "pipeline__name"]
    readonly_fields = ["id", "created_at", "updated_at"]

    # Map configuration - Nigeria center
    default_lat = 9.0820
    default_lon = 8.6753
    default_zoom = 7
    map_width = 900
    map_height = 600

    fieldsets = (
        (
            "Segment Information",
            {
                "fields": (
                    "id",
                    "pipeline",
                    "segment_name",
                    "segment_number",
                    "length_km",
                )
            },
        ),
        (
            "Geographic Information",
            {"fields": ("start_point", "end_point", "geometry")},
        ),
        ("Monitoring", {"fields": ("monitoring_frequency_hours", "last_monitored")}),
        ("Risk Assessment", {"fields": ("risk_level", "terrain_type")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(MonitoringConfiguration)
class MonitoringConfigurationAdmin(admin.ModelAdmin):
    list_display = [
        "pipeline",
        "nasa_api_enabled",
        "analysis_enabled",
        "alerts_enabled",
    ]
    list_filter = ["nasa_api_enabled", "analysis_enabled", "alerts_enabled"]
    search_fields = ["pipeline__name"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        ("Configuration", {"fields": ("id", "pipeline")}),
        ("API Settings", {"fields": ("nasa_api_key", "nasa_api_enabled")}),
        (
            "Analysis Settings",
            {"fields": ("analysis_enabled", "auto_analysis", "confidence_threshold")},
        ),
        (
            "Alert Settings",
            {"fields": ("alerts_enabled", "email_notifications", "sms_notifications")},
        ),
        (
            "Monitoring Schedule",
            {"fields": ("monitoring_frequency_hours", "analysis_frequency_hours")},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = ["action", "resource_type", "user", "success", "created_at"]
    list_filter = ["action", "resource_type", "success", "created_at"]
    search_fields = [
        "action",
        "resource_name",
        "user__email",
        "user__first_name",
        "user__last_name",
    ]
    readonly_fields = ["id", "created_at"]
    date_hierarchy = "created_at"

    fieldsets = (
        ("Activity Information", {"fields": ("id", "user", "action", "success")}),
        (
            "Resource Information",
            {"fields": ("resource_type", "resource_id", "resource_name", "details")},
        ),
        ("Request Information", {"fields": ("ip_address", "user_agent")}),
        ("Timestamp", {"fields": ("created_at",)}),
    )


@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ["site_name", "updated_at"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            "General Settings",
            {
                "fields": (
                    "site_name",
                    "site_description",
                    "timezone",
                    "language",
                    "theme",
                )
            },
        ),
        (
            "Notification Settings",
            {
                "fields": (
                    "email_notifications",
                    "sms_notifications",
                    "push_notifications",
                    "alert_email",
                    "alert_phone",
                )
            },
        ),
        (
            "Analysis Settings",
            {
                "fields": (
                    "analysis_confidence_threshold",
                    "auto_analysis_enabled",
                    "analysis_retention_days",
                )
            },
        ),
        (
            "Security Settings",
            {
                "fields": (
                    "session_timeout",
                    "password_min_length",
                    "require_2fa",
                    "failed_login_lockout",
                )
            },
        ),
        ("API Settings", {"fields": ("api_rate_limit", "api_key_expiry_days")}),
        (
            "Storage Settings",
            {
                "fields": (
                    "max_file_size_mb",
                    "storage_quota_gb",
                    "auto_cleanup_enabled",
                )
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def has_add_permission(self, request):
        # Only allow one instance
        return not SystemSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        # Don't allow deletion
        return False


# Customize admin site
admin.site.site_header = "Pipeline Monitoring System"
admin.site.site_title = "Pipeline Monitoring Admin"
admin.site.index_title = "Welcome to Flow Safe Administration"
