"""
Celery tasks for pipeline monitoring system.
"""

import logging
from datetime import datetime, timedelta
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from .models import Pipeline, SatelliteImage, AnalysisResult, MonitoringAlert
from .scripts.satellite_data_fetcher import fetch_pipeline_imagery_task
from .scripts.image_analyzer import analyze_satellite_image_task
from .scripts.anomaly_detector import detect_anomalies_task

logger = logging.getLogger(__name__)


@shared_task
def fetch_satellite_imagery_periodic():
    """
    Periodic task to fetch satellite imagery for all active pipelines
    """
    try:
        active_pipelines = Pipeline.objects.filter(status='active')
        
        for pipeline in active_pipelines:
            # Check if we need to fetch new imagery
            last_image = SatelliteImage.objects.filter(
                pipeline=pipeline
            ).order_by('-image_date').first()
            
            if last_image:
                # Only fetch if last image is older than monitoring frequency
                config = getattr(pipeline, 'monitoring_config', None)
                frequency_hours = config.monitoring_frequency_hours if config else 24
                
                if last_image.image_date < timezone.now() - timedelta(hours=frequency_hours):
                    fetch_pipeline_imagery.delay(str(pipeline.id))
            else:
                # No images yet, fetch recent imagery
                fetch_pipeline_imagery.delay(str(pipeline.id))
        
        logger.info(f"Triggered satellite imagery fetch for {active_pipelines.count()} pipelines")
        
    except Exception as e:
        logger.error(f"Error in periodic satellite imagery fetch: {e}")


@shared_task
def fetch_pipeline_imagery(pipeline_id: str, start_date: str = None, end_date: str = None):
    """
    Task to fetch satellite imagery for a specific pipeline
    """
    try:
        if not start_date:
            start_date = (timezone.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = timezone.now().strftime('%Y-%m-%d')
        
        # Call the satellite data fetcher
        fetch_pipeline_imagery_task(pipeline_id, start_date, end_date)
        
        logger.info(f"Completed satellite imagery fetch for pipeline {pipeline_id}")
        
    except Exception as e:
        logger.error(f"Error fetching satellite imagery for pipeline {pipeline_id}: {e}")


@shared_task
def analyze_images_periodic():
    """
    Periodic task to analyze pending satellite images
    """
    try:
        pending_images = SatelliteImage.objects.filter(
            processing_status='completed',
            analysis_results__isnull=True
        ).distinct()
        
        for image in pending_images:
            analyze_satellite_image.delay(str(image.id))
        
        logger.info(f"Triggered analysis for {pending_images.count()} images")
        
    except Exception as e:
        logger.error(f"Error in periodic image analysis: {e}")


@shared_task
def analyze_satellite_image(image_id: str, analysis_types: list = None):
    """
    Task to analyze a specific satellite image
    """
    try:
        if analysis_types is None:
            analysis_types = ['leak_detection', 'oil_spill', 'vandalism', 'anomaly']
        
        # Call the image analyzer
        analyze_satellite_image_task(image_id, analysis_types)
        
        logger.info(f"Completed analysis for image {image_id}")
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_id}: {e}")


@shared_task
def detect_anomalies_periodic():
    """
    Periodic task to detect anomalies across all pipelines
    """
    try:
        active_pipelines = Pipeline.objects.filter(status='active')
        
        for pipeline in active_pipelines:
            # Detect anomalies for the last 7 days
            end_date = timezone.now()
            start_date = end_date - timedelta(days=7)
            
            detect_anomalies.delay(
                str(pipeline.id),
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        logger.info(f"Triggered anomaly detection for {active_pipelines.count()} pipelines")
        
    except Exception as e:
        logger.error(f"Error in periodic anomaly detection: {e}")


@shared_task
def detect_anomalies(pipeline_id: str, start_date: str, end_date: str):
    """
    Task to detect anomalies for a specific pipeline
    """
    try:
        # Call the anomaly detector
        detect_anomalies_task(pipeline_id, start_date, end_date)
        
        logger.info(f"Completed anomaly detection for pipeline {pipeline_id}")
        
    except Exception as e:
        logger.error(f"Error detecting anomalies for pipeline {pipeline_id}: {e}")


@shared_task
def cleanup_old_data():
    """
    Periodic task to clean up old data
    """
    try:
        # Clean up old analysis results (older than 1 year)
        cutoff_date = timezone.now() - timedelta(days=365)
        old_results = AnalysisResult.objects.filter(created_at__lt=cutoff_date)
        old_count = old_results.count()
        old_results.delete()
        
        # Clean up old satellite images (older than 2 years)
        image_cutoff = timezone.now() - timedelta(days=730)
        old_images = SatelliteImage.objects.filter(created_at__lt=image_cutoff)
        image_count = old_images.count()
        old_images.delete()
        
        # Clean up resolved alerts (older than 6 months)
        alert_cutoff = timezone.now() - timedelta(days=180)
        old_alerts = MonitoringAlert.objects.filter(
            is_resolved=True,
            created_at__lt=alert_cutoff
        )
        alert_count = old_alerts.count()
        old_alerts.delete()
        
        logger.info(f"Cleanup completed: {old_count} analysis results, {image_count} images, {alert_count} alerts removed")
        
    except Exception as e:
        logger.error(f"Error in data cleanup: {e}")


@shared_task
def send_alert_notifications():
    """
    Task to send notifications for unresolved alerts
    """
    try:
        unresolved_alerts = MonitoringAlert.objects.filter(
            is_resolved=False,
            sent_at__isnull=True
        )
        
        for alert in unresolved_alerts:
            send_alert_notification.delay(str(alert.id))
        
        logger.info(f"Triggered notifications for {unresolved_alerts.count()} alerts")
        
    except Exception as e:
        logger.error(f"Error sending alert notifications: {e}")


@shared_task
def send_alert_notification(alert_id: str):
    """
    Task to send a specific alert notification
    """
    try:
        alert = MonitoringAlert.objects.get(id=alert_id)
        
        # Update sent timestamp
        alert.sent_at = timezone.now()
        alert.save()
        
        # TODO: Implement actual notification sending (email, SMS, etc.)
        logger.info(f"Alert notification sent for alert {alert_id}")
        
    except MonitoringAlert.DoesNotExist:
        logger.error(f"Alert {alert_id} not found")
    except Exception as e:
        logger.error(f"Error sending alert notification {alert_id}: {e}")


@shared_task
def generate_daily_report():
    """
    Task to generate daily monitoring report
    """
    try:
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)
        
        # Get statistics for the day
        pipelines_count = Pipeline.objects.filter(status='active').count()
        images_count = SatelliteImage.objects.filter(
            image_date__date=today
        ).count()
        analyses_count = AnalysisResult.objects.filter(
            created_at__date=today
        ).count()
        alerts_count = MonitoringAlert.objects.filter(
            created_at__date=today
        ).count()
        critical_alerts = MonitoringAlert.objects.filter(
            created_at__date=today,
            priority='urgent'
        ).count()
        
        report_data = {
            'date': today,
            'pipelines_monitored': pipelines_count,
            'images_processed': images_count,
            'analyses_completed': analyses_count,
            'alerts_generated': alerts_count,
            'critical_alerts': critical_alerts,
        }
        
        # TODO: Send report via email or store in database
        logger.info(f"Daily report generated: {report_data}")
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")


@shared_task
def update_pipeline_health_scores():
    """
    Task to update health scores for all pipelines
    """
    try:
        pipelines = Pipeline.objects.filter(status='active')
        
        for pipeline in pipelines:
            update_pipeline_health.delay(str(pipeline.id))
        
        logger.info(f"Triggered health score updates for {pipelines.count()} pipelines")
        
    except Exception as e:
        logger.error(f"Error updating pipeline health scores: {e}")


@shared_task
def update_pipeline_health(pipeline_id: str):
    """
    Task to update health score for a specific pipeline
    """
    try:
        pipeline = Pipeline.objects.get(id=pipeline_id)
        
        # Calculate health score based on recent analysis results
        recent_results = AnalysisResult.objects.filter(
            satellite_image__pipeline=pipeline,
            created_at__gte=timezone.now() - timedelta(days=30)
        )
        
        if recent_results.exists():
            # Calculate health metrics
            total_analyses = recent_results.count()
            critical_issues = recent_results.filter(severity='critical').count()
            high_issues = recent_results.filter(severity='high').count()
            
            # Calculate health score (0-100)
            health_score = 100
            health_score -= critical_issues * 20  # -20 for each critical issue
            health_score -= high_issues * 10      # -10 for each high issue
            health_score = max(0, health_score)   # Don't go below 0
            
            # TODO: Store health score in pipeline model or separate health model
            logger.info(f"Pipeline {pipeline_id} health score: {health_score}")
        
    except Pipeline.DoesNotExist:
        logger.error(f"Pipeline {pipeline_id} not found")
    except Exception as e:
        logger.error(f"Error updating health for pipeline {pipeline_id}: {e}")


@shared_task
def process_batch_images(image_ids: list, analysis_types: list = None):
    """
    Task to process multiple images in batch
    """
    try:
        if analysis_types is None:
            analysis_types = ['leak_detection', 'oil_spill', 'vandalism', 'anomaly']
        
        for image_id in image_ids:
            analyze_satellite_image.delay(image_id, analysis_types)
        
        logger.info(f"Triggered batch processing for {len(image_ids)} images")
        
    except Exception as e:
        logger.error(f"Error in batch image processing: {e}")


@shared_task
def export_monitoring_data(pipeline_id: str, start_date: str, end_date: str, format: str = 'csv'):
    """
    Task to export monitoring data for a pipeline
    """
    try:
        # TODO: Implement data export functionality
        logger.info(f"Exporting data for pipeline {pipeline_id} from {start_date} to {end_date} in {format} format")
        
    except Exception as e:
        logger.error(f"Error exporting data for pipeline {pipeline_id}: {e}")
