"""
Image Analysis Module for Pipeline Monitoring System

This module provides comprehensive image analysis capabilities for detecting
pipeline leaks, oil spills, vandalism, and other anomalies using computer vision
and machine learning techniques.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    analysis_type: str
    confidence_score: float
    severity: str
    location: Tuple[float, float]  # (lat, lon)
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    description: str
    raw_data: Dict[str, Any]
    affected_area: Optional[np.ndarray] = None


class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image for better analysis"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


class LeakDetector:
    """Detector for pipeline leaks using computer vision"""
    
    def __init__(self):
        self.leak_threshold = 0.7
        self.min_contour_area = 100
    
    def detect_leaks(self, image: np.ndarray) -> List[AnalysisResult]:
        """Detect potential pipeline leaks in image"""
        results = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define range for leak colors (dark spots, discoloration)
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            lower_leak = np.array([0, 50, 50])
            upper_leak = np.array([20, 255, 255])
            
            # Create mask for potential leak areas
            mask = cv2.inRange(hsv, lower_leak, upper_leak)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Calculate confidence based on area and shape
                    confidence = min(area / 1000, 1.0)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center point (convert to lat/lon if needed)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Determine severity based on area and confidence
                    if area > 500 and confidence > 0.8:
                        severity = 'critical'
                    elif area > 200 and confidence > 0.6:
                        severity = 'high'
                    elif area > 100 and confidence > 0.4:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    if confidence > self.leak_threshold:
                        result = AnalysisResult(
                            analysis_type='leak_detection',
                            confidence_score=confidence,
                            severity=severity,
                            location=(center_y, center_x),  # Placeholder coordinates
                            bounding_box=(x, y, w, h),
                            description=f"Potential pipeline leak detected (area: {area:.0f} pixels)",
                            raw_data={
                                'area': area,
                                'contour_points': len(contour),
                                'aspect_ratio': w / h if h > 0 else 0
                            }
                        )
                        results.append(result)
            
        except Exception as e:
            logger.error(f"Error in leak detection: {e}")
        
        return results


class OilSpillDetector:
    """Detector for oil spills using spectral analysis"""
    
    def __init__(self):
        self.spill_threshold = 0.6
        self.min_spill_area = 200
    
    def detect_oil_spills(self, image: np.ndarray) -> List[AnalysisResult]:
        """Detect potential oil spills in image"""
        results = []
        
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Oil spill detection using multiple approaches
            
            # 1. Dark spot detection (oil appears dark)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            dark_spots = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 2. Color-based detection (oil has specific spectral signature)
            # This is simplified - real implementation would use hyperspectral data
            oil_mask = self._create_oil_mask(hsv)
            
            # Combine masks
            combined_mask = cv2.bitwise_and(dark_spots, oil_mask)
            
            # Apply morphological operations
            kernel = np.ones((7, 7), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_spill_area:
                    # Calculate confidence based on area and shape characteristics
                    confidence = self._calculate_spill_confidence(contour, area)
                    
                    if confidence > self.spill_threshold:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Determine severity
                        if area > 1000 and confidence > 0.8:
                            severity = 'critical'
                        elif area > 500 and confidence > 0.6:
                            severity = 'high'
                        elif area > 200 and confidence > 0.4:
                            severity = 'medium'
                        else:
                            severity = 'low'
                        
                        result = AnalysisResult(
                            analysis_type='oil_spill',
                            confidence_score=confidence,
                            severity=severity,
                            location=(center_y, center_x),
                            bounding_box=(x, y, w, h),
                            description=f"Potential oil spill detected (area: {area:.0f} pixels)",
                            raw_data={
                                'area': area,
                                'perimeter': cv2.arcLength(contour, True),
                                'circularity': 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                            }
                        )
                        results.append(result)
            
        except Exception as e:
            logger.error(f"Error in oil spill detection: {e}")
        
        return results
    
    def _create_oil_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Create mask for oil-like colors"""
        # Oil typically appears as dark, bluish-black areas
        lower_oil = np.array([100, 50, 20])
        upper_oil = np.array([130, 255, 100])
        
        mask1 = cv2.inRange(hsv, lower_oil, upper_oil)
        
        # Also detect very dark areas
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        
        mask2 = cv2.inRange(hsv, lower_dark, upper_dark)
        
        return cv2.bitwise_or(mask1, mask2)
    
    def _calculate_spill_confidence(self, contour: np.ndarray, area: float) -> float:
        """Calculate confidence score for oil spill detection"""
        # Calculate circularity (oil spills tend to be more circular)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Combine factors
        confidence = (circularity * 0.6 + min(aspect_ratio, 1/aspect_ratio) * 0.4)
        return min(confidence, 1.0)


class VandalismDetector:
    """Detector for pipeline vandalism and unauthorized activities"""
    
    def __init__(self):
        self.vandalism_threshold = 0.5
        self.min_change_area = 150
    
    def detect_vandalism(self, current_image: np.ndarray, 
                        reference_image: np.ndarray = None) -> List[AnalysisResult]:
        """Detect potential vandalism by comparing with reference image"""
        results = []
        
        try:
            if reference_image is None:
                # If no reference image, look for suspicious patterns
                return self._detect_suspicious_patterns(current_image)
            
            # Compare with reference image
            diff = self._calculate_difference(current_image, reference_image)
            
            # Apply threshold to find significant changes
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_change_area:
                    confidence = min(area / 500, 1.0)
                    
                    if confidence > self.vandalism_threshold:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Determine severity based on change magnitude
                        change_magnitude = np.mean(diff[y:y+h, x:x+w])
                        if change_magnitude > 100:
                            severity = 'high'
                        elif change_magnitude > 50:
                            severity = 'medium'
                        else:
                            severity = 'low'
                        
                        result = AnalysisResult(
                            analysis_type='vandalism',
                            confidence_score=confidence,
                            severity=severity,
                            location=(center_y, center_x),
                            bounding_box=(x, y, w, h),
                            description=f"Potential vandalism detected (change area: {area:.0f} pixels)",
                            raw_data={
                                'area': area,
                                'change_magnitude': change_magnitude,
                                'contour_points': len(contour)
                            }
                        )
                        results.append(result)
            
        except Exception as e:
            logger.error(f"Error in vandalism detection: {e}")
        
        return results
    
    def _detect_suspicious_patterns(self, image: np.ndarray) -> List[AnalysisResult]:
        """Detect suspicious patterns without reference image"""
        results = []
        
        try:
            # Look for straight lines (potential cuts or damage)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length > 100:  # Significant line length
                        confidence = min(length / 200, 1.0)
                        
                        if confidence > self.vandalism_threshold:
                            result = AnalysisResult(
                                analysis_type='vandalism',
                                confidence_score=confidence,
                                severity='medium',
                                location=((y1+y2)//2, (x1+x2)//2),
                                bounding_box=(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)),
                                description=f"Suspicious linear pattern detected (length: {length:.0f} pixels)",
                                raw_data={
                                    'line_length': length,
                                    'angle': np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                                }
                            )
                            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in suspicious pattern detection: {e}")
        
        return results
    
    def _calculate_difference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Calculate difference between two images"""
        # Resize images to same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        return diff


class AnomalyDetector:
    """General anomaly detector using machine learning"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.is_fitted = False
    
    def detect_anomalies(self, image: np.ndarray) -> List[AnalysisResult]:
        """Detect general anomalies using ML approach"""
        results = []
        
        try:
            # Extract features from image
            features = self._extract_features(image)
            
            if not self.is_fitted:
                # For now, use a simple threshold-based approach
                # In production, you'd train on historical data
                return self._simple_anomaly_detection(image)
            
            # Normalize features
            features_scaled = self.scaler.transform([features])
            features_pca = self.pca.transform(features_scaled)
            
            # Predict anomaly
            anomaly_score = self.isolation_forest.decision_function(features_pca)[0]
            is_anomaly = self.isolation_forest.predict(features_pca)[0] == -1
            
            if is_anomaly:
                confidence = abs(anomaly_score)
                severity = 'high' if confidence > 0.5 else 'medium'
                
                result = AnalysisResult(
                    analysis_type='anomaly',
                    confidence_score=confidence,
                    severity=severity,
                    location=(image.shape[0]//2, image.shape[1]//2),
                    bounding_box=(0, 0, image.shape[1], image.shape[0]),
                    description=f"Anomaly detected (score: {anomaly_score:.3f})",
                    raw_data={
                        'anomaly_score': anomaly_score,
                        'features': features.tolist()
                    }
                )
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return results
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image for ML analysis"""
        features = []
        
        # Color features
        mean_color = np.mean(image, axis=(0, 1))
        std_color = np.std(image, axis=(0, 1))
        features.extend(mean_color)
        features.extend(std_color)
        
        # Texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern (simplified)
        lbp = self._calculate_lbp(gray)
        features.extend([np.mean(lbp), np.std(lbp)])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([np.mean(edges), np.std(edges)])
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend([np.mean(hist), np.std(hist)])
        
        return np.array(features)
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern (simplified)"""
        # This is a simplified LBP implementation
        # In production, you'd use a proper LBP library
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                binary = 0
                binary |= (image[i-1, j-1] >= center) << 7
                binary |= (image[i-1, j] >= center) << 6
                binary |= (image[i-1, j+1] >= center) << 5
                binary |= (image[i, j+1] >= center) << 4
                binary |= (image[i+1, j+1] >= center) << 3
                binary |= (image[i+1, j] >= center) << 2
                binary |= (image[i+1, j-1] >= center) << 1
                binary |= (image[i, j-1] >= center) << 0
                lbp[i, j] = binary
        
        return lbp
    
    def _simple_anomaly_detection(self, image: np.ndarray) -> List[AnalysisResult]:
        """Simple anomaly detection without ML model"""
        results = []
        
        try:
            # Calculate image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Look for unusual patterns
            if std_intensity > 50:  # High variation
                confidence = min(std_intensity / 100, 1.0)
                
                if confidence > 0.6:
                    result = AnalysisResult(
                        analysis_type='anomaly',
                        confidence_score=confidence,
                        severity='medium',
                        location=(image.shape[0]//2, image.shape[1]//2),
                        bounding_box=(0, 0, image.shape[1], image.shape[0]),
                        description=f"Unusual image variation detected (std: {std_intensity:.1f})",
                        raw_data={
                            'mean_intensity': mean_intensity,
                            'std_intensity': std_intensity
                        }
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Error in simple anomaly detection: {e}")
        
        return results


class ImageAnalyzer:
    """Main image analyzer that coordinates all detection methods"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.leak_detector = LeakDetector()
        self.oil_spill_detector = OilSpillDetector()
        self.vandalism_detector = VandalismDetector()
        self.anomaly_detector = AnomalyDetector()
    
    def analyze_image(self, image_path: str, analysis_types: List[str] = None,
                     reference_image_path: str = None) -> List[AnalysisResult]:
        """
        Analyze an image for various types of anomalies
        
        Args:
            image_path: Path to the image to analyze
            analysis_types: List of analysis types to perform
            reference_image_path: Path to reference image for comparison
            
        Returns:
            List of AnalysisResult objects
        """
        if analysis_types is None:
            analysis_types = ['leak_detection', 'oil_spill', 'vandalism', 'anomaly']
        
        # Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        if image is None:
            return []
        
        # Enhance image
        image = self.preprocessor.enhance_image(image)
        
        # Load reference image if provided
        reference_image = None
        if reference_image_path and os.path.exists(reference_image_path):
            reference_image = self.preprocessor.load_image(reference_image_path)
            if reference_image is not None:
                reference_image = self.preprocessor.enhance_image(reference_image)
        
        results = []
        
        # Perform requested analyses
        if 'leak_detection' in analysis_types:
            leak_results = self.leak_detector.detect_leaks(image)
            results.extend(leak_results)
        
        if 'oil_spill' in analysis_types:
            spill_results = self.oil_spill_detector.detect_oil_spills(image)
            results.extend(spill_results)
        
        if 'vandalism' in analysis_types:
            vandalism_results = self.vandalism_detector.detect_vandalism(image, reference_image)
            results.extend(vandalism_results)
        
        if 'anomaly' in analysis_types:
            anomaly_results = self.anomaly_detector.detect_anomalies(image)
            results.extend(anomaly_results)
        
        return results
    
    def batch_analyze(self, image_paths: List[str], analysis_types: List[str] = None) -> Dict[str, List[AnalysisResult]]:
        """
        Analyze multiple images in batch
        
        Args:
            image_paths: List of image paths to analyze
            analysis_types: List of analysis types to perform
            
        Returns:
            Dictionary mapping image paths to analysis results
        """
        results = {}
        
        for image_path in image_paths:
            try:
                analysis_results = self.analyze_image(image_path, analysis_types)
                results[image_path] = analysis_results
                logger.info(f"Analyzed {image_path}: {len(analysis_results)} results")
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {e}")
                results[image_path] = []
        
        return results


def analyze_satellite_image_task(image_id: str, analysis_types: List[str] = None):
    """
    Celery task for analyzing satellite images
    
    This function should be decorated with @celery_app.task for use with Celery
    """
    from monitoring.models import SatelliteImage, AnalysisResult
    from django.contrib.gis.geos import Point, Polygon
    from django.conf import settings
    
    try:
        # Get satellite image
        satellite_image = SatelliteImage.objects.get(id=image_id)
        
        if satellite_image.processing_status != 'completed':
            logger.error(f"Image {image_id} not ready for analysis")
            return
        
        # Initialize analyzer
        analyzer = ImageAnalyzer()
        
        # Get image path
        image_path = os.path.join(settings.MEDIA_ROOT, satellite_image.image_file.name)
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        # Perform analysis
        if analysis_types is None:
            analysis_types = ['leak_detection', 'oil_spill', 'vandalism', 'anomaly']
        
        results = analyzer.analyze_image(image_path, analysis_types)
        
        # Save results to database
        for result in results:
            # Convert pixel coordinates to geographic coordinates
            # This is a simplified conversion - in practice, you'd use proper georeferencing
            bounds = satellite_image.bounds
            lat_range = bounds.extent[3] - bounds.extent[1]  # max_lat - min_lat
            lon_range = bounds.extent[2] - bounds.extent[0]  # max_lon - min_lon
            
            # Convert pixel coordinates to lat/lon
            pixel_lat = bounds.extent[1] + (result.location[0] / satellite_image.image_file.height) * lat_range
            pixel_lon = bounds.extent[0] + (result.location[1] / satellite_image.image_file.width) * lon_range
            
            # Create analysis result record
            analysis_result = AnalysisResult.objects.create(
                satellite_image=satellite_image,
                analysis_type=result.analysis_type,
                confidence_score=result.confidence_score,
                severity=result.severity,
                detected_location=Point(pixel_lon, pixel_lat),
                description=result.description,
                raw_data=result.raw_data,
                status='pending'
            )
            
            logger.info(f"Created analysis result {analysis_result.id} for {result.analysis_type}")
        
        logger.info(f"Completed analysis for image {image_id}: {len(results)} results")
        
    except Exception as e:
        logger.error(f"Error in image analysis task: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = ImageAnalyzer()
    
    # Analyze a single image
    results = analyzer.analyze_image(
        "path/to/image.jpg",
        analysis_types=['leak_detection', 'oil_spill']
    )
    
    print(f"Found {len(results)} anomalies:")
    for result in results:
        print(f"- {result.analysis_type}: {result.confidence_score:.2f} confidence ({result.severity})")
