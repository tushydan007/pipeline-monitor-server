"""
Celery configuration for pipeline_monitoring project.
"""

import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pipeline_monitoring.settings")

app = Celery("pipeline_monitoring")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery beat schedule for periodic tasks
app.conf.beat_schedule = {
    "fetch-satellite-imagery": {
        "task": "monitoring.tasks.fetch_satellite_imagery_periodic",
        "schedule": 24 * 60 * 60,  # Run every 24 hours
    },
    "analyze-images": {
        "task": "monitoring.tasks.analyze_images_periodic",
        "schedule": 6 * 60 * 60,  # Run every 6 hours
    },
    "detect-anomalies": {
        "task": "monitoring.tasks.detect_anomalies_periodic",
        "schedule": 12 * 60 * 60,  # Run every 12 hours
    },
    "cleanup-old-data": {
        "task": "monitoring.tasks.cleanup_old_data",
        "schedule": 7 * 24 * 60 * 60,  # Run every 7 days
    },
}

app.conf.timezone = "UTC"


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
