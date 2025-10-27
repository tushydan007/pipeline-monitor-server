"""
Data seeding script for pipeline monitoring system.
Creates sample data for development and testing.
"""

import os
import sys
import django
from datetime import datetime, timedelta
from django.contrib.gis.geos import Point, LineString, Polygon
from django.utils import timezone

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pipeline_monitoring.settings')
django.setup()

from monitoring.models import (
    Pipeline, SatelliteImage, AnalysisResult, MonitoringAlert,
    PipelineSegment, MonitoringConfiguration
)
from django.contrib.auth.models import User


def create_sample_pipeline():
    """Create a sample pipeline in South-South Nigeria"""
    
    # South-South Nigeria coordinates (Port Harcourt area)
    start_point = Point(6.9981, 4.8156)  # Port Harcourt
    end_point = Point(7.1981, 4.9156)    # 30km northeast
    
    # Create pipeline route
    route = LineString([start_point, end_point])
    
    pipeline = Pipeline.objects.create(
        name="South-South Nigeria Pipeline",
        description="30km pipeline in South-South region for oil transportation",
        length_km=30.0,
        diameter_mm=500.0,
        material="steel",
        status="active",
        start_point=start_point,
        end_point=end_point,
        route=route
    )
    
    print(f"Created pipeline: {pipeline.name}")
    return pipeline


def create_pipeline_segments(pipeline):
    """Create segments for the pipeline"""
    
    # Divide 30km pipeline into 10 segments of 3km each
    start_lon, start_lat = pipeline.start_point.x, pipeline.start_point.y
    end_lon, end_lat = pipeline.end_point.x, pipeline.end_point.y
    
    segments = []
    for i in range(10):
        # Calculate segment start and end points
        segment_start_lon = start_lon + (end_lon - start_lon) * (i / 10)
        segment_start_lat = start_lat + (end_lat - start_lat) * (i / 10)
        segment_end_lon = start_lon + (end_lon - start_lon) * ((i + 1) / 10)
        segment_end_lat = start_lat + (end_lat - start_lat) * ((i + 1) / 10)
        
        segment_start = Point(segment_start_lon, segment_start_lat)
        segment_end = Point(segment_end_lon, segment_end_lat)
        segment_geometry = LineString([segment_start, segment_end])
        
        # Determine terrain type based on segment
        terrain_types = ['urban', 'rural', 'forest', 'water', 'mountain']
        terrain_type = terrain_types[i % len(terrain_types)]
        
        # Determine risk level
        risk_levels = ['low', 'medium', 'high', 'critical']
        risk_level = risk_levels[i % len(risk_levels)]
        
        segment = PipelineSegment.objects.create(
            pipeline=pipeline,
            segment_name=f"Segment {i + 1}",
            segment_number=i + 1,
            length_km=3.0,
            start_point=segment_start,
            end_point=segment_end,
            geometry=segment_geometry,
            risk_level=risk_level,
            terrain_type=terrain_type,
            monitoring_frequency_hours=24
        )
        
        segments.append(segment)
    
    print(f"Created {len(segments)} pipeline segments")
    return segments


def create_sample_images(pipeline):
    """Create sample satellite images"""
    
    images = []
    base_date = timezone.now() - timedelta(days=30)
    
    for i in range(10):
        # Create image date (every 3 days)
        image_date = base_date + timedelta(days=i * 3)
        
        # Create bounds around pipeline
        center_lon = (pipeline.start_point.x + pipeline.end_point.x) / 2
        center_lat = (pipeline.start_point.y + pipeline.end_point.y) / 2
        
        # Create bounds (0.1 degree buffer)
        bounds = Polygon.from_bbox((
            center_lon - 0.05, center_lat - 0.05,
            center_lon + 0.05, center_lat + 0.05
        ))
        
        center_point = Point(center_lon, center_lat)
        
        image = SatelliteImage.objects.create(
            pipeline=pipeline,
            image_date=image_date,
            satellite_name="Landsat-8",
            sensor="OLI_TIRS",
            resolution_m=30.0,
            bounds=bounds,
            center_point=center_point,
            source_api="nasa",
            api_image_id=f"landsat8_{i:03d}",
            processing_status="completed"
        )
        
        images.append(image)
    
    print(f"Created {len(images)} satellite images")
    return images


def create_sample_analysis_results(images):
    """Create sample analysis results"""
    
    results = []
    analysis_types = ['leak_detection', 'oil_spill', 'vandalism', 'vegetation_change']
    severities = ['low', 'medium', 'high', 'critical']
    
    for image in images:
        # Create 1-3 analysis results per image
        import random
        num_results = random.randint(1, 3)
        
        for j in range(num_results):
            analysis_type = random.choice(analysis_types)
            severity = random.choice(severities)
            confidence = random.uniform(0.3, 0.9)
            
            # Create random location within image bounds
            bounds = image.bounds
            lon = random.uniform(bounds.extent[0], bounds.extent[2])
            lat = random.uniform(bounds.extent[1], bounds.extent[3])
            detected_location = Point(lon, lat)
            
            # Create affected area (small polygon around detected location)
            affected_area = Polygon.from_bbox((
                lon - 0.001, lat - 0.001,
                lon + 0.001, lat + 0.001
            ))
            
            result = AnalysisResult.objects.create(
                satellite_image=image,
                analysis_type=analysis_type,
                confidence_score=confidence,
                severity=severity,
                detected_location=detected_location,
                affected_area=affected_area,
                description=f"Sample {analysis_type.replace('_', ' ')} detection",
                raw_data={
                    'area': random.uniform(100, 1000),
                    'confidence': confidence,
                    'severity': severity
                },
                status='pending'
            )
            
            results.append(result)
    
    print(f"Created {len(results)} analysis results")
    return results


def create_sample_alerts(results):
    """Create sample monitoring alerts"""
    
    alerts = []
    priorities = ['low', 'medium', 'high', 'urgent']
    
    for result in results:
        if result.severity in ['high', 'critical']:
            priority = 'urgent' if result.severity == 'critical' else 'high'
            
            alert = MonitoringAlert.objects.create(
                analysis_result=result,
                alert_type=f"{result.analysis_type}_alert",
                priority=priority,
                message=f"Alert: {result.analysis_type.replace('_', ' ')} detected with {result.confidence_score:.1%} confidence",
                is_resolved=False
            )
            
            alerts.append(alert)
    
    print(f"Created {len(alerts)} monitoring alerts")
    return alerts


def create_monitoring_configuration(pipeline):
    """Create monitoring configuration for pipeline"""
    
    config = MonitoringConfiguration.objects.create(
        pipeline=pipeline,
        nasa_api_key="sample-api-key",
        nasa_api_enabled=True,
        analysis_enabled=True,
        auto_analysis=True,
        confidence_threshold=0.7,
        alerts_enabled=True,
        email_notifications=True,
        sms_notifications=False,
        monitoring_frequency_hours=24,
        analysis_frequency_hours=6
    )
    
    print(f"Created monitoring configuration for {pipeline.name}")
    return config


def create_sample_user():
    """Create a sample user for testing"""
    
    user, created = User.objects.get_or_create(
        username='admin',
        defaults={
            'email': 'admin@pipeline-monitoring.com',
            'first_name': 'Admin',
            'last_name': 'User',
            'is_staff': True,
            'is_superuser': True
        }
    )
    
    if created:
        user.set_password('admin123')
        user.save()
        print("Created admin user: admin/admin123")
    else:
        print("Admin user already exists")
    
    return user


def main():
    """Main seeding function"""
    
    print("Starting data seeding...")
    
    # Create sample user
    user = create_sample_user()
    
    # Create sample pipeline
    pipeline = create_sample_pipeline()
    
    # Create pipeline segments
    segments = create_pipeline_segments(pipeline)
    
    # Create sample images
    images = create_sample_images(pipeline)
    
    # Create analysis results
    results = create_sample_analysis_results(images)
    
    # Create alerts
    alerts = create_sample_alerts(results)
    
    # Create monitoring configuration
    config = create_monitoring_configuration(pipeline)
    
    print("\nData seeding completed!")
    print(f"Summary:")
    print(f"- 1 pipeline created")
    print(f"- {len(segments)} segments created")
    print(f"- {len(images)} satellite images created")
    print(f"- {len(results)} analysis results created")
    print(f"- {len(alerts)} alerts created")
    print(f"- 1 monitoring configuration created")
    print(f"- 1 admin user created (username: admin, password: admin123)")


if __name__ == "__main__":
    main()
