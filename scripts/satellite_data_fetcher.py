"""
Satellite Data Fetcher for Pipeline Monitoring System

This module handles fetching satellite imagery from various sources including NASA APIs.
It provides functionality to retrieve satellite data for pipeline monitoring areas.
"""

import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from django.conf import settings
from django.utils import timezone
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class SatelliteImageData:
    """Data class for satellite image information"""

    image_id: str
    satellite_name: str
    sensor: str
    image_date: datetime
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    resolution_m: float
    image_url: str
    thumbnail_url: Optional[str] = None
    cloud_cover: Optional[float] = None
    quality_score: Optional[float] = None


class NASASatelliteFetcher:
    """Fetcher for NASA satellite imagery"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nasa.gov"
        self.earth_api_url = f"{self.base_url}/planetary/earth/imagery"
        self.earth_assets_url = f"{self.base_url}/planetary/earth/assets"

    def get_imagery(
        self,
        lat: float,
        lon: float,
        date: str,
        dim: float = 0.15,
        cloud_score: bool = True,
    ) -> Optional[Dict]:
        """
        Get satellite imagery from NASA API

        Args:
            lat: Latitude
            lon: Longitude
            date: Date in YYYY-MM-DD format
            dim: Dimension of the image in degrees (default 0.15)
            cloud_score: Whether to include cloud score

        Returns:
            Dictionary containing image data or None if failed
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "date": date,
                "dim": dim,
                "api_key": self.api_key,
            }

            if cloud_score:
                params["cloud_score"] = "True"

            response = requests.get(self.earth_api_url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"NASA API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching NASA imagery: {e}")
            return None

    def get_assets(
        self, lat: float, lon: float, begin_date: str, end_date: str
    ) -> Optional[List[Dict]]:
        """
        Get available satellite assets for a location and date range

        Args:
            lat: Latitude
            lon: Longitude
            begin_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of available assets or None if failed
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "begin": begin_date,
                "end": end_date,
                "api_key": self.api_key,
            }

            response = requests.get(self.earth_assets_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data.get("results", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"NASA Assets API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching NASA assets: {e}")
            return None


class LandsatFetcher:
    """Fetcher for Landsat satellite imagery"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.nasa.gov"
        # Note: This would typically use USGS Landsat API or similar

    def search_scenes(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
    ) -> List[Dict]:
        """
        Search for Landsat scenes within bounds and date range

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_cloud_cover: Maximum cloud cover percentage

        Returns:
            List of scene information
        """
        # This is a placeholder implementation
        # In a real implementation, you would use USGS Landsat API
        logger.info(
            f"Searching Landsat scenes for bounds {bounds} from {start_date} to {end_date}"
        )
        return []


class SentinelFetcher:
    """Fetcher for Sentinel satellite imagery"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        # Note: This would typically use Copernicus Open Access Hub API

    def search_products(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        product_type: str = "S2MSI1C",
    ) -> List[Dict]:
        """
        Search for Sentinel products within bounds and date range

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            product_type: Sentinel product type

        Returns:
            List of product information
        """
        # This is a placeholder implementation
        logger.info(
            f"Searching Sentinel products for bounds {bounds} from {start_date} to {end_date}"
        )
        return []


class SatelliteDataManager:
    """Main manager for satellite data fetching and processing"""

    def __init__(self, nasa_api_key: str):
        self.nasa_fetcher = NASASatelliteFetcher(nasa_api_key)
        self.landsat_fetcher = LandsatFetcher()
        self.sentinel_fetcher = SentinelFetcher()

    def fetch_pipeline_imagery(
        self,
        pipeline_id: str,
        start_date: str,
        end_date: str,
        sources: List[str] = None,
    ) -> List[SatelliteImageData]:
        """
        Fetch satellite imagery for a pipeline

        Args:
            pipeline_id: UUID of the pipeline
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sources: List of sources to use ('nasa', 'landsat', 'sentinel')

        Returns:
            List of SatelliteImageData objects
        """
        if sources is None:
            sources = ["nasa"]

        # Get pipeline information
        from monitoring.models import Pipeline

        try:
            pipeline = Pipeline.objects.get(id=pipeline_id)
        except Pipeline.DoesNotExist:
            logger.error(f"Pipeline {pipeline_id} not found")
            return []

        # Calculate pipeline bounds
        bounds = self._calculate_pipeline_bounds(pipeline)
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        images = []

        # Fetch from NASA
        if "nasa" in sources:
            nasa_images = self._fetch_nasa_imagery(
                center_lat, center_lon, start_date, end_date
            )
            images.extend(nasa_images)

        # Fetch from Landsat
        if "landsat" in sources:
            landsat_images = self._fetch_landsat_imagery(bounds, start_date, end_date)
            images.extend(landsat_images)

        # Fetch from Sentinel
        if "sentinel" in sources:
            sentinel_images = self._fetch_sentinel_imagery(bounds, start_date, end_date)
            images.extend(sentinel_images)

        return images

    def _calculate_pipeline_bounds(self, pipeline) -> Tuple[float, float, float, float]:
        """Calculate bounds for a pipeline with buffer"""
        buffer_degrees = 0.01  # ~1km buffer

        start_lon, start_lat = pipeline.start_point.x, pipeline.start_point.y
        end_lon, end_lat = pipeline.end_point.x, pipeline.end_point.y

        min_lon = min(start_lon, end_lon) - buffer_degrees
        max_lon = max(start_lon, end_lon) + buffer_degrees
        min_lat = min(start_lat, end_lat) - buffer_degrees
        max_lat = max(start_lat, end_lat) + buffer_degrees

        return (min_lon, min_lat, max_lon, max_lat)

    def _fetch_nasa_imagery(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> List[SatelliteImageData]:
        """Fetch NASA imagery for a location and date range"""
        images = []

        # Get available assets first
        assets = self.nasa_fetcher.get_assets(lat, lon, start_date, end_date)
        if not assets:
            return images

        # Process each asset
        for asset in assets:
            try:
                image_data = self.nasa_fetcher.get_imagery(
                    lat, lon, asset["date"], cloud_score=True
                )

                if image_data and "url" in image_data:
                    # Calculate bounds (approximate)
                    dim = 0.15  # Default dimension
                    bounds = (
                        lon - dim / 2,
                        lat - dim / 2,
                        lon + dim / 2,
                        lat + dim / 2,
                    )

                    image_info = SatelliteImageData(
                        image_id=asset.get("id", f"nasa_{asset['date']}"),
                        satellite_name="Landsat",
                        sensor="OLI_TIRS",
                        image_date=datetime.strptime(asset["date"], "%Y-%m-%d"),
                        bounds=bounds,
                        resolution_m=30.0,  # Landsat resolution
                        image_url=image_data["url"],
                        cloud_cover=image_data.get("cloud_score"),
                        quality_score=1.0 - (image_data.get("cloud_score", 0) or 0),
                    )

                    images.append(image_info)

            except Exception as e:
                logger.error(f"Error processing NASA asset {asset}: {e}")
                continue

        return images

    def _fetch_landsat_imagery(
        self, bounds: Tuple[float, float, float, float], start_date: str, end_date: str
    ) -> List[SatelliteImageData]:
        """Fetch Landsat imagery for bounds and date range"""
        # Placeholder implementation
        return []

    def _fetch_sentinel_imagery(
        self, bounds: Tuple[float, float, float, float], start_date: str, end_date: str
    ) -> List[SatelliteImageData]:
        """Fetch Sentinel imagery for bounds and date range"""
        # Placeholder implementation
        return []

    def download_and_process_image(
        self, image_data: SatelliteImageData, output_path: str
    ) -> bool:
        """
        Download and process a satellite image

        Args:
            image_data: SatelliteImageData object
            output_path: Path to save the processed image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Download image
            response = requests.get(image_data.image_url, timeout=60)
            response.raise_for_status()

            # Process image
            with Image.open(io.BytesIO(response.content)) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if too large
                max_size = (2048, 2048)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save processed image
                img.save(output_path, "JPEG", quality=85)

            return True

        except Exception as e:
            logger.error(f"Error downloading/processing image: {e}")
            return False

    def create_thumbnail(
        self, image_path: str, thumbnail_path: str, size: Tuple[int, int] = (300, 300)
    ) -> bool:
        """
        Create a thumbnail from an image

        Args:
            image_path: Path to source image
            thumbnail_path: Path to save thumbnail
            size: Thumbnail size

        Returns:
            True if successful, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "JPEG", quality=80)
            return True
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return False


def fetch_pipeline_imagery_task(
    pipeline_id: str, start_date: str, end_date: str, sources: List[str] = None
):
    """
    Celery task for fetching pipeline imagery

    This function should be decorated with @celery_app.task for use with Celery
    """
    from monitoring.models import Pipeline, SatelliteImage
    from django.contrib.gis.geos import Polygon, Point

    # Get NASA API key from settings
    nasa_api_key = getattr(settings, "NASA_API_KEY", "")
    if not nasa_api_key:
        logger.error("NASA API key not configured")
        return

    manager = SatelliteDataManager(nasa_api_key)
    images = manager.fetch_pipeline_imagery(pipeline_id, start_date, end_date, sources)

    pipeline = Pipeline.objects.get(id=pipeline_id)

    for image_data in images:
        try:
            # Create satellite image record
            satellite_image = SatelliteImage.objects.create(
                pipeline=pipeline,
                image_date=image_data.image_date,
                satellite_name=image_data.satellite_name,
                sensor=image_data.sensor,
                resolution_m=image_data.resolution_m,
                bounds=Polygon.from_bbox(image_data.bounds),
                center_point=Point(
                    (image_data.bounds[0] + image_data.bounds[2]) / 2,
                    (image_data.bounds[1] + image_data.bounds[3]) / 2,
                ),
                source_api="nasa",
                api_image_id=image_data.image_id,
                processing_status="pending",
            )

            # Download and process image
            image_filename = f"{satellite_image.id}.jpg"
            image_path = os.path.join(
                settings.MEDIA_ROOT, "satellite_images", image_filename
            )
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            if manager.download_and_process_image(image_data, image_path):
                satellite_image.image_file.name = f"satellite_images/{image_filename}"

                # Create thumbnail
                thumbnail_filename = f"thumb_{satellite_image.id}.jpg"
                thumbnail_path = os.path.join(
                    settings.MEDIA_ROOT, "thumbnails", thumbnail_filename
                )
                os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)

                if manager.create_thumbnail(image_path, thumbnail_path):
                    satellite_image.thumbnail.name = f"thumbnails/{thumbnail_filename}"

                satellite_image.processing_status = "completed"
                satellite_image.save()

                logger.info(f"Successfully processed image {satellite_image.id}")
            else:
                satellite_image.processing_status = "failed"
                satellite_image.save()
                logger.error(f"Failed to process image {satellite_image.id}")

        except Exception as e:
            logger.error(f"Error creating satellite image record: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    nasa_api_key = "your_nasa_api_key_here"
    manager = SatelliteDataManager(nasa_api_key)

    # Fetch imagery for a pipeline
    images = manager.fetch_pipeline_imagery(
        pipeline_id="your_pipeline_id",
        start_date="2024-01-01",
        end_date="2024-01-31",
        sources=["nasa"],
    )

    print(f"Found {len(images)} images")
    for img in images:
        print(
            f"Image: {img.satellite_name} - {img.image_date} - Quality: {img.quality_score}"
        )
