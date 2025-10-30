from django.contrib import admin
from django.contrib.gis.admin import GISModelAdmin
from django.utils.html import format_html
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


@admin.register(Pipeline)
class PipelineAdmin(GISModelAdmin):
    list_display = ["name", "length_km", "material", "status", "created_at"]
    list_filter = ["status", "material", "created_at"]
    search_fields = ["name", "description"]
    readonly_fields = ["id", "created_at", "updated_at"]

    # Map configuration - Port Harcourt, Nigeria (Niger Delta region)
    default_lat = 4.8156
    default_lon = 7.0498
    default_zoom = 11
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
            "Geographic Information",
            {
                "fields": ("start_point", "end_point", "route"),
                "description": "These fields are optional. You can add geographic data later.",
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


@admin.register(AnalysisResult)
class AnalysisResultAdmin(GISModelAdmin):
    list_display = ["confidence_score", "severity", "status", "created_at"]
    list_filter = ["severity", "status", "created_at"]
    search_fields = ["description"]
    readonly_fields = ["id", "created_at", "updated_at"]

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
admin.site.index_title = "Welcome to Pipeline Monitoring Administration"
