"""
Advanced Anomaly Detection Module for Pipeline Monitoring System

This module provides sophisticated anomaly detection capabilities using machine learning
techniques including time series analysis, spatial clustering, and ensemble methods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Data class for anomaly detection results"""
    anomaly_type: str
    confidence_score: float
    severity: str
    location: Tuple[float, float]
    timestamp: datetime
    description: str
    features: Dict[str, Any]
    raw_data: Dict[str, Any]


class TimeSeriesAnomalyDetector:
    """Detector for time series anomalies in pipeline monitoring data"""
    
    def __init__(self, window_size: int = 24, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        value_column: str = 'value',
                        timestamp_column: str = 'timestamp') -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in time series data
        
        Args:
            data: DataFrame with time series data
            value_column: Name of the value column
            timestamp_column: Name of the timestamp column
            
        Returns:
            List of detected anomalies
        """
        results = []
        
        try:
            # Sort by timestamp
            data = data.sort_values(timestamp_column)
            
            # Extract values
            values = data[value_column].values
            timestamps = data[timestamp_column].values
            
            # Apply multiple detection methods
            statistical_anomalies = self._detect_statistical_anomalies(values, timestamps)
            trend_anomalies = self._detect_trend_anomalies(values, timestamps)
            seasonal_anomalies = self._detect_seasonal_anomalies(values, timestamps)
            
            # Combine results
            all_anomalies = statistical_anomalies + trend_anomalies + seasonal_anomalies
            
            # Remove duplicates and merge similar anomalies
            merged_anomalies = self._merge_anomalies(all_anomalies)
            
            results.extend(merged_anomalies)
            
        except Exception as e:
            logger.error(f"Error in time series anomaly detection: {e}")
        
        return results
    
    def _detect_statistical_anomalies(self, values: np.ndarray, 
                                    timestamps: np.ndarray) -> List[AnomalyDetectionResult]:
        """Detect statistical outliers using Z-score and IQR methods"""
        anomalies = []
        
        try:
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_anomalies = np.where(z_scores > self.threshold)[0]
            
            for idx in z_anomalies:
                confidence = min(z_scores[idx] / 4.0, 1.0)  # Normalize to 0-1
                severity = 'high' if z_scores[idx] > 3.0 else 'medium'
                
                anomaly = AnomalyDetectionResult(
                    anomaly_type='statistical_outlier',
                    confidence_score=confidence,
                    severity=severity,
                    location=(0, 0),  # Placeholder
                    timestamp=timestamps[idx],
                    description=f"Statistical outlier detected (Z-score: {z_scores[idx]:.2f})",
                    features={'z_score': z_scores[idx], 'value': values[idx]},
                    raw_data={'method': 'z_score', 'threshold': self.threshold}
                )
                anomalies.append(anomaly)
            
            # IQR method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
            
            for idx in iqr_anomalies:
                distance = min(abs(values[idx] - lower_bound), abs(values[idx] - upper_bound))
                confidence = min(distance / IQR, 1.0)
                severity = 'high' if distance > 2 * IQR else 'medium'
                
                anomaly = AnomalyDetectionResult(
                    anomaly_type='statistical_outlier',
                    confidence_score=confidence,
                    severity=severity,
                    location=(0, 0),  # Placeholder
                    timestamp=timestamps[idx],
                    description=f"IQR outlier detected (value: {values[idx]:.2f})",
                    features={'value': values[idx], 'iqr': IQR, 'bounds': [lower_bound, upper_bound]},
                    raw_data={'method': 'iqr', 'Q1': Q1, 'Q3': Q3}
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    def _detect_trend_anomalies(self, values: np.ndarray, 
                               timestamps: np.ndarray) -> List[AnomalyDetectionResult]:
        """Detect anomalies in trends using moving averages and derivatives"""
        anomalies = []
        
        try:
            # Calculate moving average
            window = min(self.window_size, len(values) // 4)
            if window < 3:
                return anomalies
            
            moving_avg = pd.Series(values).rolling(window=window, center=True).mean()
            
            # Calculate residuals
            residuals = values - moving_avg.fillna(0)
            
            # Detect significant deviations from trend
            residual_std = np.std(residuals)
            trend_anomalies = np.where(np.abs(residuals) > 2 * residual_std)[0]
            
            for idx in trend_anomalies:
                if not np.isnan(moving_avg.iloc[idx]):
                    deviation = abs(residuals[idx]) / residual_std
                    confidence = min(deviation / 4.0, 1.0)
                    severity = 'high' if deviation > 3.0 else 'medium'
                    
                    anomaly = AnomalyDetectionResult(
                        anomaly_type='trend_anomaly',
                        confidence_score=confidence,
                        severity=severity,
                        location=(0, 0),  # Placeholder
                        timestamp=timestamps[idx],
                        description=f"Trend anomaly detected (deviation: {deviation:.2f}σ)",
                        features={
                            'value': values[idx],
                            'expected': moving_avg.iloc[idx],
                            'residual': residuals[idx],
                            'deviation_sigma': deviation
                        },
                        raw_data={'method': 'trend_analysis', 'window_size': window}
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in trend anomaly detection: {e}")
        
        return anomalies
    
    def _detect_seasonal_anomalies(self, values: np.ndarray, 
                                  timestamps: np.ndarray) -> List[AnomalyDetectionResult]:
        """Detect anomalies in seasonal patterns"""
        anomalies = []
        
        try:
            # Convert to pandas for easier time series operations
            df = pd.DataFrame({'value': values, 'timestamp': timestamps})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Detect daily seasonality
            daily_pattern = df.groupby(df.index.hour)['value'].mean()
            daily_std = df.groupby(df.index.hour)['value'].std()
            
            for idx, row in df.iterrows():
                hour = idx.hour
                expected_value = daily_pattern[hour]
                expected_std = daily_std[hour]
                
                if not np.isnan(expected_std) and expected_std > 0:
                    z_score = abs(row['value'] - expected_value) / expected_std
                    
                    if z_score > self.threshold:
                        confidence = min(z_score / 4.0, 1.0)
                        severity = 'high' if z_score > 3.0 else 'medium'
                        
                        anomaly = AnomalyDetectionResult(
                            anomaly_type='seasonal_anomaly',
                            confidence_score=confidence,
                            severity=severity,
                            location=(0, 0),  # Placeholder
                            timestamp=idx,
                            description=f"Seasonal anomaly detected (hour {hour}, Z-score: {z_score:.2f})",
                            features={
                                'value': row['value'],
                                'expected': expected_value,
                                'hour': hour,
                                'z_score': z_score
                            },
                            raw_data={'method': 'seasonal_analysis', 'pattern': 'daily'}
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in seasonal anomaly detection: {e}")
        
        return anomalies
    
    def _merge_anomalies(self, anomalies: List[AnomalyDetectionResult]) -> List[AnomalyDetectionResult]:
        """Merge similar anomalies that are close in time"""
        if not anomalies:
            return []
        
        # Sort by timestamp
        anomalies.sort(key=lambda x: x.timestamp)
        
        merged = []
        current_anomaly = anomalies[0]
        
        for next_anomaly in anomalies[1:]:
            # If anomalies are within 1 hour and of the same type, merge them
            time_diff = abs((next_anomaly.timestamp - current_anomaly.timestamp).total_seconds())
            
            if (time_diff < 3600 and 
                next_anomaly.anomaly_type == current_anomaly.anomaly_type):
                # Merge anomalies
                current_anomaly.confidence_score = max(
                    current_anomaly.confidence_score, 
                    next_anomaly.confidence_score
                )
                current_anomaly.description += f" (merged with {next_anomaly.timestamp})"
            else:
                merged.append(current_anomaly)
                current_anomaly = next_anomaly
        
        merged.append(current_anomaly)
        return merged


class SpatialAnomalyDetector:
    """Detector for spatial anomalies in pipeline monitoring data"""
    
    def __init__(self, eps: float = 0.1, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        lat_column: str = 'latitude',
                        lon_column: str = 'longitude',
                        value_column: str = 'value') -> List[AnomalyDetectionResult]:
        """
        Detect spatial anomalies using clustering and density-based methods
        
        Args:
            data: DataFrame with spatial data
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            value_column: Name of value column
            
        Returns:
            List of detected spatial anomalies
        """
        results = []
        
        try:
            # Prepare spatial features
            coords = data[[lat_column, lon_column]].values
            values = data[value_column].values
            
            # Normalize coordinates
            coords_scaled = self.scaler.fit_transform(coords)
            
            # Apply DBSCAN clustering
            cluster_labels = self.dbscan.fit_predict(coords_scaled)
            
            # Identify noise points (anomalies)
            noise_mask = cluster_labels == -1
            noise_indices = np.where(noise_mask)[0]
            
            for idx in noise_indices:
                # Calculate isolation score
                distances = pdist([coords_scaled[idx]], coords_scaled)
                min_distance = np.min(distances)
                isolation_score = min_distance / self.eps
                
                confidence = min(isolation_score, 1.0)
                severity = 'high' if isolation_score > 2.0 else 'medium'
                
                anomaly = AnomalyDetectionResult(
                    anomaly_type='spatial_outlier',
                    confidence_score=confidence,
                    severity=severity,
                    location=(coords[idx, 0], coords[idx, 1]),
                    timestamp=datetime.now(),  # Placeholder
                    description=f"Spatial outlier detected (isolation: {isolation_score:.2f})",
                    features={
                        'latitude': coords[idx, 0],
                        'longitude': coords[idx, 1],
                        'value': values[idx],
                        'isolation_score': isolation_score
                    },
                    raw_data={'method': 'dbscan', 'eps': self.eps, 'min_samples': self.min_samples}
                )
                results.append(anomaly)
            
            # Detect cluster-based anomalies
            cluster_anomalies = self._detect_cluster_anomalies(coords, values, cluster_labels)
            results.extend(cluster_anomalies)
            
        except Exception as e:
            logger.error(f"Error in spatial anomaly detection: {e}")
        
        return results
    
    def _detect_cluster_anomalies(self, coords: np.ndarray, values: np.ndarray, 
                                 cluster_labels: np.ndarray) -> List[AnomalyDetectionResult]:
        """Detect anomalies within clusters"""
        anomalies = []
        
        try:
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_coords = coords[cluster_mask]
                cluster_values = values[cluster_mask]
                
                if len(cluster_values) < 3:
                    continue
                
                # Calculate cluster statistics
                cluster_mean = np.mean(cluster_values)
                cluster_std = np.std(cluster_values)
                
                # Find outliers within cluster
                if cluster_std > 0:
                    z_scores = np.abs((cluster_values - cluster_mean) / cluster_std)
                    outlier_mask = z_scores > 2.0
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    for idx in outlier_indices:
                        global_idx = np.where(cluster_mask)[0][idx]
                        confidence = min(z_scores[idx] / 4.0, 1.0)
                        severity = 'high' if z_scores[idx] > 3.0 else 'medium'
                        
                        anomaly = AnomalyDetectionResult(
                            anomaly_type='cluster_outlier',
                            confidence_score=confidence,
                            severity=severity,
                            location=(coords[global_idx, 0], coords[global_idx, 1]),
                            timestamp=datetime.now(),  # Placeholder
                            description=f"Cluster outlier detected (Z-score: {z_scores[idx]:.2f})",
                            features={
                                'latitude': coords[global_idx, 0],
                                'longitude': coords[global_idx, 1],
                                'value': cluster_values[idx],
                                'cluster_id': cluster_id,
                                'cluster_mean': cluster_mean,
                                'z_score': z_scores[idx]
                            },
                            raw_data={'method': 'cluster_analysis', 'cluster_size': len(cluster_values)}
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in cluster anomaly detection: {e}")
        
        return anomalies


class EnsembleAnomalyDetector:
    """Ensemble anomaly detector combining multiple methods"""
    
    def __init__(self):
        self.time_series_detector = TimeSeriesAnomalyDetector()
        self.spatial_detector = SpatialAnomalyDetector()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        lat_column: str = 'latitude',
                        lon_column: str = 'longitude',
                        value_column: str = 'value',
                        timestamp_column: str = 'timestamp') -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using ensemble of methods
        
        Args:
            data: DataFrame with monitoring data
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            value_column: Name of value column
            timestamp_column: Name of timestamp column
            
        Returns:
            List of detected anomalies
        """
        results = []
        
        try:
            # Time series anomalies
            ts_anomalies = self.time_series_detector.detect_anomalies(
                data, value_column, timestamp_column
            )
            results.extend(ts_anomalies)
            
            # Spatial anomalies
            spatial_anomalies = self.spatial_detector.detect_anomalies(
                data, lat_column, lon_column, value_column
            )
            results.extend(spatial_anomalies)
            
            # ML-based anomalies
            ml_anomalies = self._detect_ml_anomalies(data, lat_column, lon_column, value_column)
            results.extend(ml_anomalies)
            
            # Combine and rank results
            final_results = self._combine_results(results)
            
        except Exception as e:
            logger.error(f"Error in ensemble anomaly detection: {e}")
            final_results = results
        
        return final_results
    
    def _detect_ml_anomalies(self, data: pd.DataFrame, 
                            lat_column: str, lon_column: str, 
                            value_column: str) -> List[AnomalyDetectionResult]:
        """Detect anomalies using machine learning methods"""
        anomalies = []
        
        try:
            # Prepare features
            features = data[[lat_column, lon_column, value_column]].values
            
            if not self.is_fitted:
                # Fit the model
                features_scaled = self.scaler.fit_transform(features)
                self.isolation_forest.fit(features_scaled)
                self.is_fitted = True
            
            # Predict anomalies
            features_scaled = self.scaler.transform(features)
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            is_anomaly = self.isolation_forest.predict(features_scaled) == -1
            
            # Process results
            for idx, (is_anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
                if is_anom:
                    confidence = abs(score)
                    severity = 'high' if confidence > 0.5 else 'medium'
                    
                    anomaly = AnomalyDetectionResult(
                        anomaly_type='ml_anomaly',
                        confidence_score=confidence,
                        severity=severity,
                        location=(data.iloc[idx][lat_column], data.iloc[idx][lon_column]),
                        timestamp=datetime.now(),  # Placeholder
                        description=f"ML anomaly detected (score: {score:.3f})",
                        features={
                            'latitude': data.iloc[idx][lat_column],
                            'longitude': data.iloc[idx][lon_column],
                            'value': data.iloc[idx][value_column],
                            'anomaly_score': score
                        },
                        raw_data={'method': 'isolation_forest', 'contamination': 0.1}
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
        
        return anomalies
    
    def _combine_results(self, results: List[AnomalyDetectionResult]) -> List[AnomalyDetectionResult]:
        """Combine and rank anomaly detection results"""
        if not results:
            return []
        
        # Group by location and timestamp
        grouped = defaultdict(list)
        for result in results:
            key = (result.location, result.timestamp.date())
            grouped[key].append(result)
        
        # Combine results for same location/time
        final_results = []
        for group in grouped.values():
            if len(group) == 1:
                final_results.append(group[0])
            else:
                # Combine multiple detections
                combined = self._merge_anomaly_group(group)
                final_results.append(combined)
        
        # Sort by confidence score
        final_results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return final_results
    
    def _merge_anomaly_group(self, group: List[AnomalyDetectionResult]) -> AnomalyDetectionResult:
        """Merge a group of similar anomalies"""
        # Use the highest confidence result as base
        base = max(group, key=lambda x: x.confidence_score)
        
        # Update description to include all methods
        methods = [r.anomaly_type for r in group]
        base.description = f"Anomaly detected using {', '.join(set(methods))} methods"
        
        # Update confidence (average of all methods)
        base.confidence_score = np.mean([r.confidence_score for r in group])
        
        # Update severity (use highest severity)
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        base.severity = max(group, key=lambda x: severity_levels.get(x.severity, 0)).severity
        
        # Combine features
        all_features = {}
        for result in group:
            all_features.update(result.features)
        base.features = all_features
        
        return base


class AnomalyDetectionManager:
    """Main manager for anomaly detection operations"""
    
    def __init__(self):
        self.ensemble_detector = EnsembleAnomalyDetector()
        self.detection_history = []
    
    def detect_pipeline_anomalies(self, pipeline_id: str, 
                                 start_date: datetime, 
                                 end_date: datetime) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies for a specific pipeline
        
        Args:
            pipeline_id: UUID of the pipeline
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of detected anomalies
        """
        try:
            # Get monitoring data for the pipeline
            from monitoring.models import AnalysisResult, SatelliteImage
            
            # Get analysis results for the pipeline
            analysis_results = AnalysisResult.objects.filter(
                satellite_image__pipeline_id=pipeline_id,
                created_at__gte=start_date,
                created_at__lte=end_date
            ).select_related('satellite_image')
            
            if not analysis_results.exists():
                logger.warning(f"No analysis results found for pipeline {pipeline_id}")
                return []
            
            # Convert to DataFrame
            data = []
            for result in analysis_results:
                data.append({
                    'latitude': result.detected_location.y,
                    'longitude': result.detected_location.x,
                    'value': result.confidence_score,
                    'timestamp': result.created_at,
                    'analysis_type': result.analysis_type,
                    'severity': result.severity
                })
            
            df = pd.DataFrame(data)
            
            if df.empty:
                return []
            
            # Detect anomalies
            anomalies = self.ensemble_detector.detect_anomalies(df)
            
            # Store detection history
            self.detection_history.append({
                'pipeline_id': pipeline_id,
                'timestamp': datetime.now(),
                'anomaly_count': len(anomalies)
            })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting pipeline anomalies: {e}")
            return []
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies"""
        if not self.detection_history:
            return {}
        
        total_detections = sum(entry['anomaly_count'] for entry in self.detection_history)
        avg_anomalies = total_detections / len(self.detection_history)
        
        return {
            'total_detections': total_detections,
            'average_anomalies_per_detection': avg_anomalies,
            'detection_count': len(self.detection_history),
            'last_detection': max(entry['timestamp'] for entry in self.detection_history)
        }


def detect_anomalies_task(pipeline_id: str, start_date: str, end_date: str):
    """
    Celery task for anomaly detection
    
    This function should be decorated with @celery_app.task for use with Celery
    """
    from monitoring.models import MonitoringAlert, AnalysisResult
    from django.contrib.gis.geos import Point
    
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Initialize detector
        manager = AnomalyDetectionManager()
        
        # Detect anomalies
        anomalies = manager.detect_pipeline_anomalies(pipeline_id, start_dt, end_dt)
        
        # Create alerts for high-confidence anomalies
        for anomaly in anomalies:
            if anomaly.confidence_score > 0.7:
                # Find related analysis result
                analysis_result = AnalysisResult.objects.filter(
                    satellite_image__pipeline_id=pipeline_id,
                    detected_location__distance_lte=Point(anomaly.location[1], anomaly.location[0]),
                    created_at__gte=start_dt,
                    created_at__lte=end_dt
                ).first()
                
                if analysis_result:
                    # Create alert
                    alert = MonitoringAlert.objects.create(
                        analysis_result=analysis_result,
                        alert_type=f"{anomaly.anomaly_type}_alert",
                        priority=anomaly.severity,
                        message=anomaly.description,
                        is_resolved=False
                    )
                    
                    logger.info(f"Created alert {alert.id} for anomaly {anomaly.anomaly_type}")
        
        logger.info(f"Completed anomaly detection for pipeline {pipeline_id}: {len(anomalies)} anomalies")
        
    except Exception as e:
        logger.error(f"Error in anomaly detection task: {e}")


if __name__ == "__main__":
    # Example usage
    manager = AnomalyDetectionManager()
    
    # Create sample data
    data = pd.DataFrame({
        'latitude': np.random.normal(5.0, 0.1, 100),
        'longitude': np.random.normal(6.0, 0.1, 100),
        'value': np.random.normal(0.5, 0.1, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    # Add some anomalies
    data.loc[50, 'value'] = 2.0  # Statistical outlier
    data.loc[75, 'latitude'] = 5.5  # Spatial outlier
    
    # Detect anomalies
    anomalies = manager.ensemble_detector.detect_anomalies(data)
    
    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"- {anomaly.anomaly_type}: {anomaly.confidence_score:.2f} confidence ({anomaly.severity})")
