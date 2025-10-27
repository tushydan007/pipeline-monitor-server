import django_filters
from django.db.models import Q
from .models import AnalysisResult, MonitoringAlert


class AnalysisResultFilter(django_filters.FilterSet):
    """Filter for AnalysisResult model"""
    analysis_type = django_filters.ChoiceFilter(choices=AnalysisResult.ANALYSIS_TYPE_CHOICES)
    severity = django_filters.ChoiceFilter(choices=AnalysisResult.SEVERITY_CHOICES)
    status = django_filters.ChoiceFilter(choices=AnalysisResult.STATUS_CHOICES)
    confidence_min = django_filters.NumberFilter(field_name='confidence_score', lookup_expr='gte')
    confidence_max = django_filters.NumberFilter(field_name='confidence_score', lookup_expr='lte')
    date_from = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    date_to = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    pipeline = django_filters.UUIDFilter(field_name='satellite_image__pipeline')
    verified = django_filters.BooleanFilter(method='filter_verified')
    
    class Meta:
        model = AnalysisResult
        fields = ['analysis_type', 'severity', 'status', 'confidence_min', 'confidence_max', 'date_from', 'date_to', 'pipeline', 'verified']
    
    def filter_verified(self, queryset, name, value):
        """Filter by verification status"""
        if value:
            return queryset.filter(verified_by__isnull=False)
        else:
            return queryset.filter(verified_by__isnull=True)


class MonitoringAlertFilter(django_filters.FilterSet):
    """Filter for MonitoringAlert model"""
    alert_type = django_filters.ChoiceFilter(choices=MonitoringAlert.ALERT_TYPE_CHOICES)
    priority = django_filters.ChoiceFilter(choices=MonitoringAlert.PRIORITY_CHOICES)
    is_resolved = django_filters.BooleanFilter()
    date_from = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    date_to = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    pipeline = django_filters.UUIDFilter(field_name='analysis_result__satellite_image__pipeline')
    
    class Meta:
        model = MonitoringAlert
        fields = ['alert_type', 'priority', 'is_resolved', 'date_from', 'date_to', 'pipeline']
