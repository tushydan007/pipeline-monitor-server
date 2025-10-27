from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PipelineViewSet,
    SatelliteImageViewSet,
    AnalysisResultViewSet,
    MonitoringAlertViewSet,
    PipelineSegmentViewSet,
    MonitoringConfigurationViewSet,
    DashboardViewSet,
    ActivityLogViewSet,
    SystemSettingsViewSet,
)

router = DefaultRouter()
router.register("pipelines", PipelineViewSet)
router.register("satellite-images", SatelliteImageViewSet)
router.register("analysis-results", AnalysisResultViewSet)
router.register("alerts", MonitoringAlertViewSet)
router.register("segments", PipelineSegmentViewSet)
router.register("configurations", MonitoringConfigurationViewSet)
router.register("dashboard", DashboardViewSet, basename="dashboard")
router.register("activity-logs", ActivityLogViewSet)
router.register("settings", SystemSettingsViewSet, basename="settings")

urlpatterns = [
    path("", include(router.urls)),
]
