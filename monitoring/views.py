from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Max
from django.utils import timezone
from django.contrib.auth import get_user_model
from datetime import timedelta

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
from .serializers import (
    PipelineSerializer,
    SatelliteImageSerializer,
    AnalysisResultSerializer,
    MonitoringAlertSerializer,
    PipelineSegmentSerializer,
    MonitoringConfigurationSerializer,
    DashboardStatsSerializer,
    ActivityLogSerializer,
    SystemSettingsSerializer,
)
from .filters import AnalysisResultFilter, MonitoringAlertFilter
from user.serializers import UserSerializer  # Import from users app


class PipelineViewSet(viewsets.ModelViewSet):
    """ViewSet for Pipeline model"""

    queryset = Pipeline.objects.all()
    serializer_class = PipelineSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    search_fields = ["name", "description", "material"]
    filterset_fields = ["status", "material"]
    ordering_fields = ["name", "length_km", "created_at"]
    ordering = ["-created_at"]

    @action(detail=True, methods=["get"])
    def segments(self, request, pk=None):
        """Get all segments for a pipeline"""
        pipeline = self.get_object()
        segments = pipeline.segments.all()
        serializer = PipelineSegmentSerializer(segments, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"])
    def recent_images(self, request, pk=None):
        """Get recent satellite images for a pipeline"""
        pipeline = self.get_object()
        days = int(request.query_params.get("days", 30))
        since = timezone.now() - timedelta(days=days)

        images = pipeline.satellite_images.filter(image_date__gte=since).order_by(
            "-image_date"
        )

        serializer = SatelliteImageSerializer(images, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"])
    def analysis_summary(self, request, pk=None):
        """Get analysis summary for a pipeline"""
        pipeline = self.get_object()

        # Get analysis results for this pipeline
        analysis_results = AnalysisResult.objects.filter(
            satellite_image__pipeline=pipeline
        )

        summary = {
            "total_analyses": analysis_results.count(),
            "leak_detections": analysis_results.filter(
                analysis_type="leak_detection"
            ).count(),
            "oil_spills": analysis_results.filter(analysis_type="oil_spill").count(),
            "vandalism": analysis_results.filter(analysis_type="vandalism").count(),
            "critical_alerts": analysis_results.filter(severity="critical").count(),
            "pending_review": analysis_results.filter(status="pending").count(),
        }

        return Response(summary)


class SatelliteImageViewSet(viewsets.ModelViewSet):
    """ViewSet for SatelliteImage model"""

    queryset = SatelliteImage.objects.all()
    serializer_class = SatelliteImageSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    search_fields = ["satellite_name", "sensor"]
    filterset_fields = ["pipeline", "satellite_name", "processing_status", "source_api"]
    ordering_fields = ["image_date", "created_at"]
    ordering = ["-image_date"]

    @action(detail=True, methods=["post"])
    def analyze(self, request, pk=None):
        """Trigger analysis for a satellite image"""
        image = self.get_object()

        if image.processing_status != "completed":
            return Response(
                {"error": "Image processing not completed yet"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # TODO: Trigger analysis task
        # This would typically queue a Celery task

        return Response({"message": "Analysis triggered successfully"})

    @action(detail=True, methods=["get"])
    def analysis_results(self, request, pk=None):
        """Get analysis results for a satellite image"""
        image = self.get_object()
        results = image.analysis_results.all()
        serializer = AnalysisResultSerializer(results, many=True)
        return Response(serializer.data)


class AnalysisResultViewSet(viewsets.ModelViewSet):
    """ViewSet for AnalysisResult model"""

    queryset = AnalysisResult.objects.all()
    serializer_class = AnalysisResultSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_class = AnalysisResultFilter
    search_fields = ["analysis_type", "description"]
    ordering_fields = ["confidence_score", "created_at", "severity"]
    ordering = ["-created_at"]

    @action(detail=True, methods=["post"])
    def verify(self, request, pk=None):
        """Verify an analysis result"""
        result = self.get_object()
        verification_notes = request.data.get("verification_notes", "")

        result.status = "confirmed"
        result.verified_by = request.user
        result.verified_at = timezone.now()
        result.verification_notes = verification_notes
        result.save()

        return Response({"message": "Analysis result verified"})

    @action(detail=True, methods=["post"])
    def mark_false_positive(self, request, pk=None):
        """Mark an analysis result as false positive"""
        result = self.get_object()
        verification_notes = request.data.get("verification_notes", "")

        result.status = "false_positive"
        result.verified_by = request.user
        result.verified_at = timezone.now()
        result.verification_notes = verification_notes
        result.save()

        return Response({"message": "Analysis result marked as false positive"})

    @action(detail=True, methods=["post"])
    def create_alert(self, request, pk=None):
        """Create an alert for an analysis result"""
        result = self.get_object()

        alert_data = {
            "analysis_result": result.id,
            "alert_type": f"{result.analysis_type}_alert",
            "priority": result.severity,
            "message": f"{result.analysis_type.replace('_', ' ').title()} detected with {result.confidence_score:.1%} confidence",
        }

        serializer = MonitoringAlertSerializer(data=alert_data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MonitoringAlertViewSet(viewsets.ModelViewSet):
    """ViewSet for MonitoringAlert model"""

    queryset = MonitoringAlert.objects.all()
    serializer_class = MonitoringAlertSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_class = MonitoringAlertFilter
    search_fields = ["message", "alert_type"]
    ordering_fields = ["priority", "created_at"]
    ordering = ["-created_at"]

    @action(detail=True, methods=["post"])
    def resolve(self, request, pk=None):
        """Resolve an alert"""
        alert = self.get_object()
        alert.is_resolved = True
        alert.save()

        return Response({"message": "Alert resolved"})

    @action(detail=False, methods=["get"])
    def active(self, request):
        """Get active (unresolved) alerts"""
        active_alerts = self.queryset.filter(is_resolved=False)
        serializer = self.get_serializer(active_alerts, many=True)
        return Response(serializer.data)

    def get_queryset(self):
        """Add filtering support"""
        queryset = super().get_queryset()

        # Filter by is_resolved if provided
        is_resolved = self.request.query_params.get("is_resolved", None)
        if is_resolved is not None:
            is_resolved = is_resolved.lower() == "true"
            queryset = queryset.filter(is_resolved=is_resolved)

        return queryset


class PipelineSegmentViewSet(viewsets.ModelViewSet):
    """ViewSet for PipelineSegment model"""

    queryset = PipelineSegment.objects.all()
    serializer_class = PipelineSegmentSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    search_fields = ["segment_name"]
    filterset_fields = ["pipeline", "risk_level", "terrain_type"]
    ordering_fields = ["segment_number", "risk_level", "last_monitored"]
    ordering = ["pipeline", "segment_number"]

    @action(detail=True, methods=["post"])
    def update_monitoring(self, request, pk=None):
        """Update last monitoring time for a segment"""
        segment = self.get_object()
        segment.last_monitored = timezone.now()
        segment.save()

        return Response({"message": "Monitoring time updated"})


class MonitoringConfigurationViewSet(viewsets.ModelViewSet):
    """ViewSet for MonitoringConfiguration model"""

    queryset = MonitoringConfiguration.objects.all()
    serializer_class = MonitoringConfigurationSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    search_fields = ["pipeline__name"]


class DashboardViewSet(viewsets.ViewSet):
    """ViewSet for dashboard data"""

    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["get"])
    def stats(self, request):
        """Get dashboard statistics"""
        # Basic counts
        total_pipelines = Pipeline.objects.count()
        active_pipelines = Pipeline.objects.filter(status="active").count()
        total_images = SatelliteImage.objects.count()
        pending_analyses = AnalysisResult.objects.filter(status="pending").count()
        active_alerts = MonitoringAlert.objects.filter(is_resolved=False).count()
        critical_alerts = MonitoringAlert.objects.filter(
            is_resolved=False, priority="urgent"
        ).count()

        # Recent anomalies (last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_anomalies = AnalysisResult.objects.filter(
            created_at__gte=week_ago, severity__in=["high", "critical"]
        ).count()

        # Last analysis date
        last_analysis = AnalysisResult.objects.aggregate(last_date=Max("created_at"))[
            "last_date"
        ]

        stats_data = {
            "total_pipelines": total_pipelines,
            "active_pipelines": active_pipelines,
            "total_images": total_images,
            "pending_analyses": pending_analyses,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "recent_anomalies": recent_anomalies,
            "last_analysis_date": last_analysis,
        }

        serializer = DashboardStatsSerializer(stats_data)
        return Response(serializer.data)

    @action(detail=False, methods=["get"], url_path="recent-activity")
    def recent_activity(self, request):
        """Get recent activity data"""
        # Recent analysis results
        recent_results = AnalysisResult.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).order_by("-created_at")[:10]

        # Recent alerts
        recent_alerts = MonitoringAlert.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).order_by("-created_at")[:10]

        # Recent satellite images
        recent_images = SatelliteImage.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).order_by("-created_at")[:10]

        activity_data = {
            "recent_results": AnalysisResultSerializer(recent_results, many=True).data,
            "recent_alerts": MonitoringAlertSerializer(recent_alerts, many=True).data,
            "recent_images": SatelliteImageSerializer(recent_images, many=True).data,
        }

        return Response(activity_data)

    @action(detail=False, methods=["get"], url_path="pipeline-health")
    def pipeline_health(self, request):
        """Get pipeline health overview"""
        pipelines = Pipeline.objects.all()

        health_data = []
        for pipeline in pipelines:
            # Get recent analysis results
            recent_results = AnalysisResult.objects.filter(
                satellite_image__pipeline=pipeline,
                created_at__gte=timezone.now() - timedelta(days=30),
            )

            # Calculate health metrics
            total_analyses = recent_results.count()
            critical_issues = recent_results.filter(severity="critical").count()
            high_issues = recent_results.filter(severity="high").count()

            # Determine health status
            if critical_issues > 0:
                health_status = "critical"
            elif high_issues > 2:
                health_status = "warning"
            elif total_analyses == 0:
                health_status = "unknown"
            else:
                health_status = "healthy"

            health_data.append(
                {
                    "pipeline_id": str(pipeline.id),
                    "pipeline_name": pipeline.name,
                    "health_status": health_status,
                    "total_analyses": total_analyses,
                    "critical_issues": critical_issues,
                    "high_issues": high_issues,
                    "last_analysis": (
                        recent_results.first().created_at
                        if recent_results.exists()
                        else None
                    ),
                }
            )

        return Response(health_data)


class ActivityLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for ActivityLog model (Read-only)"""

    queryset = ActivityLog.objects.all()
    serializer_class = ActivityLogSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    search_fields = ["action", "resource_type", "resource_name"]
    filterset_fields = ["action", "resource_type", "user", "success"]
    ordering_fields = ["created_at", "action", "user"]
    ordering = ["-created_at"]


class SystemSettingsViewSet(viewsets.ModelViewSet):
    """ViewSet for SystemSettings model"""

    queryset = SystemSettings.objects.all()
    serializer_class = SystemSettingsSerializer
    permission_classes = [IsAdminUser]

    def get_object(self):
        """Return the only system settings instance"""
        obj, created = SystemSettings.objects.get_or_create()
        return obj

    def list(self, request, *args, **kwargs):
        """Return the settings object as a list"""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response([serializer.data])

    @action(detail=False, methods=["put"])
    def update_settings(self, request):
        """Update system settings without requiring ID"""
        instance, created = SystemSettings.objects.get_or_create()
        serializer = self.get_serializer(instance, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def current(self, request):
        """Get current system settings"""
        instance, created = SystemSettings.objects.get_or_create()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
