from django import forms
from django.contrib.gis.geos import GEOSGeometry, Point, LineString
from .models import Pipeline
import json
import logging
from math import radians, cos, sin, asin, sqrt

logger = logging.getLogger(__name__)


class PipelineGeoJSONUploadForm(forms.ModelForm):
    """Custom form for Pipeline with GeoJSON upload capability"""
    
    geojson_file = forms.FileField(
        required=False,
        label="Upload GeoJSON File",
        help_text="Upload a GeoJSON file containing a LineString geometry. The file can include properties like 'name', 'description', 'length_km', 'diameter_mm', 'material', and 'status'. If properties are present, they will override form fields.",
        widget=forms.FileInput(attrs={
            'accept': '.json,.geojson',
            'class': 'fileinput'
        })
    )
    
    manual_coordinates = forms.CharField(
        required=False,
        label="Manual Coordinate Entry",
        help_text="Enter coordinates manually. Formats accepted:\n"
                  "1. MultiLineString JSON (recommended): {\"type\": \"MultiLineString\", \"coordinates\": [[[lon, lat], ...], [[lon, lat], ...], ...]}\n"
                  "2. LineString JSON: [[lon1, lat1], [lon2, lat2], ...]\n"
                  "3. One per line: lon1,lat1\\nlon2,lat2\\n...\n"
                  "4. Semicolon separated: lon1,lat1;lon2,lat2;...\n"
                  "Coordinates should be in [longitude, latitude] format (GeoJSON standard).\n"
                  "For MultiLineString, all segments will be combined into a single continuous route.",
        widget=forms.Textarea(attrs={
            'rows': 15,
            'cols': 100,
            'placeholder': '{"type": "MultiLineString", "coordinates": [[[8.0082895, 4.5597778], [8.0083045, 4.5596568]], [[8.0071383, 4.5597250], [8.0082895, 4.5597778]]]}',
            'class': 'vLargeTextField',
            'style': 'font-family: monospace; font-size: 12px;'
        })
    )
    
    class Meta:
        model = Pipeline
        fields = '__all__'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make geographic fields optional when GeoJSON is uploaded
        self.fields['start_point'].required = False
        self.fields['end_point'].required = False
        self.fields['route'].required = False
    
    def clean_geojson_file(self):
        """Validate and parse GeoJSON file"""
        geojson_file = self.cleaned_data.get('geojson_file')
        
        if geojson_file:
            try:
                # Read file content
                file_content = geojson_file.read()
                
                # Try to decode as UTF-8
                try:
                    geojson_data = json.loads(file_content.decode('utf-8'))
                except UnicodeDecodeError:
                    raise forms.ValidationError("GeoJSON file must be UTF-8 encoded JSON.")
                
                # Validate GeoJSON structure
                if not isinstance(geojson_data, dict):
                    raise forms.ValidationError("GeoJSON file must contain a valid GeoJSON object.")
                
                # Handle FeatureCollection
                if geojson_data.get('type') == 'FeatureCollection':
                    features = geojson_data.get('features', [])
                    if not features:
                        raise forms.ValidationError("GeoJSON FeatureCollection is empty.")
                    # Use first feature
                    feature = features[0]
                # Handle Feature
                elif geojson_data.get('type') == 'Feature':
                    feature = geojson_data
                # Handle direct Geometry
                elif geojson_data.get('type') in ['LineString', 'MultiLineString']:
                    feature = {
                        'type': 'Feature',
                        'geometry': geojson_data,
                        'properties': {}
                    }
                else:
                    raise forms.ValidationError(
                        f"GeoJSON type '{geojson_data.get('type')}' not supported. "
                        "Expected Feature, FeatureCollection, or LineString geometry."
                    )
                
                # Validate geometry type
                geometry = feature.get('geometry', {})
                geometry_type = geometry.get('type')
                
                if geometry_type == 'LineString':
                    coordinates = geometry.get('coordinates', [])
                    if len(coordinates) < 2:
                        raise forms.ValidationError("LineString must have at least 2 coordinates.")
                elif geometry_type == 'MultiLineString':
                    # Flatten all LineString segments from MultiLineString into one continuous route
                    multi_coords = geometry.get('coordinates', [])
                    if not multi_coords or not isinstance(multi_coords, list):
                        raise forms.ValidationError("MultiLineString must contain at least one LineString segment.")
                    
                    # Connect segments intelligently: only connect if segments share an endpoint
                    coordinates = []
                    prev_end = None
                    
                    for segment_idx, segment in enumerate(multi_coords):
                        if not isinstance(segment, list) or len(segment) < 1:
                            continue
                        
                        segment_start = segment[0]
                        segment_end = segment[-1]
                        
                        # Check if this segment connects to the previous one
                        if prev_end is not None:
                            # Check if start of current segment matches end of previous (with tolerance)
                            tolerance = 1e-6
                            start_matches = (
                                abs(prev_end[0] - segment_start[0]) < tolerance and
                                abs(prev_end[1] - segment_start[1]) < tolerance
                            )
                            
                            if not start_matches:
                                # Segments don't connect - don't add the segment start yet
                                # We'll add it naturally when processing this segment
                                pass
                        
                        # Add all coordinates from this segment
                        for coord in segment:
                            # Avoid adding duplicate coordinates at connection points
                            if len(coordinates) == 0:
                                coordinates.append(coord)
                            else:
                                last_coord = coordinates[-1]
                                # Only add if it's different from the last coordinate
                                if not (
                                    abs(last_coord[0] - coord[0]) < 1e-10 and
                                    abs(last_coord[1] - coord[1]) < 1e-10
                                ):
                                    coordinates.append(coord)
                        
                        prev_end = segment_end
                    
                    if len(coordinates) < 2:
                        raise forms.ValidationError("MultiLineString must contain at least 2 coordinates total across all segments.")
                    
                    # Update geometry to be LineString with flattened coordinates
                    geometry = {
                        'type': 'LineString',
                        'coordinates': coordinates
                    }
                    feature['geometry'] = geometry
                    logger.info(f"Flattened MultiLineString with {len(multi_coords)} segments into {len(coordinates)} coordinates")
                else:
                    raise forms.ValidationError(
                        f"Geometry type '{geometry_type}' not supported. "
                        "Expected LineString or MultiLineString."
                    )
                
                # Store parsed data for use in save method
                self.geojson_data = {
                    'geometry': geometry,
                    'properties': feature.get('properties', {})
                }
                
                return geojson_file
                
            except json.JSONDecodeError as e:
                raise forms.ValidationError(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                raise forms.ValidationError(f"Error parsing GeoJSON: {str(e)}")
        
        return geojson_file
    
    def clean_manual_coordinates(self):
        """Validate and parse manual coordinate entry - supports LineString and MultiLineString formats"""
        manual_coords = self.cleaned_data.get('manual_coordinates')
        if not manual_coords or not manual_coords.strip():
            return None
        
        def validate_coordinate(coord, index=None):
            """Helper to validate a single coordinate"""
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                idx_str = f" at index {index}" if index is not None else ""
                raise forms.ValidationError(
                    f"Invalid coordinate format{idx_str}: {coord}. Expected [longitude, latitude]"
                )
            if not isinstance(coord[0], (int, float)) or not isinstance(coord[1], (int, float)):
                idx_str = f" at index {index}" if index is not None else ""
                raise forms.ValidationError(
                    f"Coordinate values must be numbers{idx_str}: {coord}"
                )
            if not (-180 <= coord[0] <= 180):
                raise forms.ValidationError(
                    f"Longitude must be between -180 and 180: {coord[0]}"
                )
            if not (-90 <= coord[1] <= 90):
                raise forms.ValidationError(
                    f"Latitude must be between -90 and 90: {coord[1]}"
                )
            return [float(coord[0]), float(coord[1])]
        
        try:
            coordinates = []
            coords_str = manual_coords.strip()
            
            # Try to parse as JSON first
            try:
                parsed = json.loads(coords_str)
                
                # Handle different JSON object formats
                if isinstance(parsed, dict):
                    geom_type = parsed.get("type")
                    coords_data = None
                    actual_geom_type = None
                    
                    # Handle FeatureCollection format
                    if geom_type == "FeatureCollection":
                        features = parsed.get("features", [])
                        if not features:
                            raise forms.ValidationError("FeatureCollection is empty")
                        
                        # Use first feature
                        feature = features[0]
                        geometry = feature.get("geometry", {})
                        actual_geom_type = geometry.get("type")
                        coords_data = geometry.get("coordinates", [])
                        
                    # Handle Feature format
                    elif geom_type == "Feature":
                        geometry = parsed.get("geometry", {})
                        actual_geom_type = geometry.get("type")
                        coords_data = geometry.get("coordinates", [])
                    
                    # Handle direct geometry formats (LineString, MultiLineString)
                    elif geom_type in ["LineString", "MultiLineString"]:
                        actual_geom_type = geom_type
                        coords_data = parsed.get("coordinates", [])
                    
                    # Now process based on actual geometry type
                    if actual_geom_type == "MultiLineString":
                        if not isinstance(coords_data, list):
                            raise forms.ValidationError("MultiLineString coordinates must be an array")
                        
                        # Flatten all LineString segments into one continuous route
                        # Only connect segments that share an endpoint to avoid drawing straight lines
                        prev_end = None
                        for segment_idx, segment in enumerate(coords_data):
                            if not isinstance(segment, list):
                                raise forms.ValidationError(
                                    f"MultiLineString segment {segment_idx} must be an array"
                                )
                            
                            if len(segment) == 0:
                                continue
                            
                            segment_start = segment[0]
                            segment_end = segment[-1]
                            
                            # Check if this segment connects to the previous one
                            if prev_end is not None:
                                # Check if start of current segment matches end of previous (with tolerance)
                                tolerance = 1e-6
                                start_matches = (
                                    abs(prev_end[0] - segment_start[0]) < tolerance and
                                    abs(prev_end[1] - segment_start[1]) < tolerance
                                )
                            
                            # Add coordinates, avoiding duplicates at connection points
                            for coord_idx, coord in enumerate(segment):
                                validated_coord = validate_coordinate(coord, f"{segment_idx}.{coord_idx}")
                                
                                if len(coordinates) == 0:
                                    coordinates.append(validated_coord)
                                else:
                                    last_coord = coordinates[-1]
                                    # Only add if it's different from the last coordinate (avoid duplicates)
                                    if not (
                                        abs(last_coord[0] - validated_coord[0]) < 1e-10 and
                                        abs(last_coord[1] - validated_coord[1]) < 1e-10
                                    ):
                                        coordinates.append(validated_coord)
                            
                            prev_end = segment_end
                        
                        if len(coordinates) < 2:
                            raise forms.ValidationError(
                                "MultiLineString must contain at least 2 coordinates total across all segments"
                            )
                        
                        logger.info(f"Parsed MultiLineString with {len(coords_data)} segments, {len(coordinates)} total coordinates")
                        
                    elif actual_geom_type == "LineString":
                        if not isinstance(coords_data, list):
                            raise forms.ValidationError("LineString coordinates must be an array")
                        
                        for idx, coord in enumerate(coords_data):
                            validated_coord = validate_coordinate(coord, idx)
                            coordinates.append(validated_coord)
                    
                    elif coords_data is None:
                        # Try fallback: check if coordinates exist at root level
                        coords_data = parsed.get("coordinates", [])
                        if coords_data and isinstance(coords_data, list) and len(coords_data) > 0:
                            # Check if it's nested (MultiLineString structure)
                            if isinstance(coords_data[0], list) and isinstance(coords_data[0][0], list):
                                # MultiLineString structure
                                for segment_idx, segment in enumerate(coords_data):
                                    for coord_idx, coord in enumerate(segment):
                                        validated_coord = validate_coordinate(coord, f"{segment_idx}.{coord_idx}")
                                        coordinates.append(validated_coord)
                            else:
                                # LineString structure
                                for idx, coord in enumerate(coords_data):
                                    validated_coord = validate_coordinate(coord, idx)
                                    coordinates.append(validated_coord)
                        else:
                            raise forms.ValidationError(
                                f"Unsupported geometry type: {geom_type}. Supported types: FeatureCollection, Feature, LineString, MultiLineString"
                            )
                    else:
                        raise forms.ValidationError(
                            f"Unsupported geometry type: {actual_geom_type or geom_type}. Supported types: FeatureCollection, Feature, LineString, MultiLineString"
                        )
                
                # Handle simple array format (LineString coordinates)
                elif isinstance(parsed, list):
                    # Check if it's a nested array (MultiLineString without wrapper)
                    if len(parsed) > 0 and isinstance(parsed[0], list) and isinstance(parsed[0][0], list):
                        # It's a MultiLineString structure [[[...], ...], [[...], ...]]
                        for segment_idx, segment in enumerate(parsed):
                            if not isinstance(segment, list):
                                raise forms.ValidationError(
                                    f"Segment {segment_idx} must be an array of coordinates"
                                )
                            for coord_idx, coord in enumerate(segment):
                                validated_coord = validate_coordinate(coord, f"{segment_idx}.{coord_idx}")
                                coordinates.append(validated_coord)
                    else:
                        # Simple LineString format [[lon, lat], [lon, lat], ...]
                        for idx, coord in enumerate(parsed):
                            validated_coord = validate_coordinate(coord, idx)
                            coordinates.append(validated_coord)
                else:
                    raise forms.ValidationError(
                        "Manual coordinates must be a JSON object (with type) or JSON array"
                    )
            except json.JSONDecodeError:
                # Not JSON, try parsing as text format
                # Handle semicolon or newline separated
                if ';' in coords_str:
                    lines = coords_str.split(';')
                else:
                    lines = coords_str.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try comma separated
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                if -180 <= lon <= 180 and -90 <= lat <= 90:
                                    coordinates.append([lon, lat])
                                else:
                                    raise forms.ValidationError(
                                        f"Invalid coordinate range: [{lon}, {lat}]"
                                    )
                            except ValueError:
                                raise forms.ValidationError(
                                    f"Invalid coordinate format in line: {line}"
                                )
                        else:
                            raise forms.ValidationError(
                                f"Each coordinate must have longitude and latitude: {line}"
                            )
                    else:
                        raise forms.ValidationError(
                            f"Could not parse coordinate: {line}. Expected format: longitude,latitude"
                        )
            
            if len(coordinates) < 2:
                raise forms.ValidationError(
                    "At least 2 coordinates are required to create a pipeline route"
                )
            
            # Store parsed coordinates for use in save method
            self.manual_coords_data = coordinates
            return manual_coords
            
        except forms.ValidationError:
            raise
        except Exception as e:
            raise forms.ValidationError(f"Error parsing manual coordinates: {str(e)}")
    
    def save(self, commit=True):
        """Override save to populate geographic fields from manual coordinates or GeoJSON"""
        instance = super().save(commit=False)
        
        # Check for manual coordinates first (takes precedence)
        manual_coords = self.cleaned_data.get('manual_coordinates')
        if manual_coords and manual_coords.strip() and hasattr(self, 'manual_coords_data'):
            try:
                coordinates = self.manual_coords_data
                
                # Create LineString directly from all coordinates
                # Coordinates are in [longitude, latitude] format
                line_coords = [(coord[0], coord[1]) for coord in coordinates]
                
                # Create LineString with all coordinates to preserve the actual route
                route_geometry = LineString(line_coords, srid=4326)
                
                # Assign the complete LineString to the route
                instance.route = route_geometry
                
                # Extract start and end points from the coordinates
                if len(coordinates) >= 2:
                    # First point: coordinates[0] is [longitude, latitude]
                    instance.start_point = Point(coordinates[0][0], coordinates[0][1], srid=4326)
                    # Last point
                    instance.end_point = Point(coordinates[-1][0], coordinates[-1][1], srid=4326)
                
                # Automatically calculate and set satellite image bounds from the route
                default_bounds = instance.calculate_default_bounds(buffer_km=1.0)
                if default_bounds:
                    instance.default_bounds = default_bounds
                    logger.info(f"Calculated default bounds for pipeline {instance.name} from manual coordinates")
                
                # Calculate length from geometry
                geometry = route_geometry
                if not instance.length_km or instance.length_km == 0:
                    instance.length_km = self._calculate_length_km(geometry)
                
            except Exception as e:
                logger.error(f"Error processing manual coordinates: {str(e)}")
                raise forms.ValidationError(f"Error processing manual coordinates: {str(e)}")
        
        # Fallback to GeoJSON file processing if no manual coordinates
        elif self.cleaned_data.get('geojson_file') and hasattr(self, 'geojson_data'):
            try:
                # Get coordinates directly from GeoJSON to ensure all points are preserved
                geojson_geometry = self.geojson_data['geometry']
                coordinates = geojson_geometry.get('coordinates', [])
                
                if len(coordinates) < 2:
                    raise ValueError("LineString must have at least 2 coordinates")
                
                # Create LineString directly from all coordinates
                # GeoJSON coordinates are [longitude, latitude]
                # Convert to tuples for LineString: (longitude, latitude) = (x, y)
                line_coords = [(coord[0], coord[1]) for coord in coordinates]
                
                # Create LineString with all coordinates to preserve the actual route
                route_geometry = LineString(line_coords, srid=4326)
                
                # Assign the complete LineString to the route
                instance.route = route_geometry
                
                # Extract start and end points from the coordinates
                if len(coordinates) >= 2:
                    # First point: coordinates[0] is [longitude, latitude]
                    instance.start_point = Point(coordinates[0][0], coordinates[0][1], srid=4326)
                    # Last point
                    instance.end_point = Point(coordinates[-1][0], coordinates[-1][1], srid=4326)
                
                    # Automatically calculate and set satellite image bounds from the route
                    # Calculate bounds with a 1km buffer around the route
                    default_bounds = instance.calculate_default_bounds(buffer_km=1.0)
                    if default_bounds:
                        instance.default_bounds = default_bounds
                        logger.info(f"Calculated default bounds for pipeline {instance.name} from GeoJSON route")
                    
                    # Use the created geometry for length calculation
                    geometry = route_geometry
                
                # Extract properties if present
                properties = self.geojson_data.get('properties', {})
                
                # Update fields from properties if not already set in form
                if properties.get('name') and not instance.name:
                    instance.name = properties['name']
                if properties.get('description') and not instance.description:
                    instance.description = properties['description']
                if properties.get('length_km') is not None:
                    try:
                        length = float(properties['length_km'])
                        if length > 0:
                            instance.length_km = length
                        else:
                            # Calculate length from geometry if not provided
                            instance.length_km = self._calculate_length_km(geometry)
                    except (ValueError, TypeError):
                        instance.length_km = self._calculate_length_km(geometry)
                elif not instance.length_km or instance.length_km == 0:
                    # Calculate length from geometry if not set
                    instance.length_km = self._calculate_length_km(geometry)
                
                if properties.get('diameter_mm') is not None:
                    try:
                        diameter = float(properties['diameter_mm'])
                        if diameter >= 10:  # Minimum validation
                            instance.diameter_mm = diameter
                    except (ValueError, TypeError):
                        pass
                
                if properties.get('material') in dict(Pipeline._meta.get_field('material').choices):
                    instance.material = properties['material']
                
                if properties.get('status') in dict(Pipeline._meta.get_field('status').choices):
                    instance.status = properties['status']
                
            except Exception as e:
                # If there's an error processing GeoJSON, allow save to continue
                # but don't populate geographic fields
                # The form will still save with manually entered data
                # Store error message for debugging
                logger.warning(f"Error processing GeoJSON file: {str(e)}")
                # Note: In a production environment, you might want to show this error
                # to the user via messages framework in the admin
        
        if commit:
            instance.save()
        return instance
    
    def _calculate_length_km(self, geometry):
        """Calculate length in kilometers from geometry"""
        if isinstance(geometry, LineString):
            # Transform to a projection that uses meters (e.g., UTM)
            # For Nigeria, we can use EPSG:32632 (UTM Zone 32N)
            try:
                geometry.transform(32632)
                length_meters = geometry.length
                length_km = length_meters / 1000.0
                # Transform back to WGS84
                geometry.transform(4326)
                return round(length_km, 2)
            except:
                # Fallback: approximate calculation using Haversine
                coords = list(geometry.coords)
                total_length = 0.0
                for i in range(len(coords) - 1):
                    lat1, lon1 = coords[i][1], coords[i][0]
                    lat2, lon2 = coords[i+1][1], coords[i+1][0]
                    
                    # Haversine formula (approximate)
                    R = 6371  # Earth radius in km
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    total_length += R * c
                
                return round(total_length, 2)
        return 0.0




# SECOND TO LATEST
# from django import forms
# from django.contrib.gis.geos import GEOSGeometry, Point, LineString
# from .models import Pipeline
# import json
# import logging
# from math import radians, cos, sin, asin, sqrt

# logger = logging.getLogger(__name__)


# class PipelineGeoJSONUploadForm(forms.ModelForm):
#     """Custom form for Pipeline with GeoJSON upload capability"""

#     geojson_file = forms.FileField(
#         required=False,
#         label="Upload GeoJSON File",
#         help_text="Upload a GeoJSON file containing a LineString geometry. The file can include properties like 'name', 'description', 'length_km', 'diameter_mm', 'material', and 'status'. If properties are present, they will override form fields.",
#         widget=forms.FileInput(attrs={
#             'accept': '.json,.geojson',
#             'class': 'fileinput'
#         })
#     )

#     class Meta:
#         model = Pipeline
#         fields = '__all__'

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Make geographic fields optional when GeoJSON is uploaded
#         self.fields['start_point'].required = False
#         self.fields['end_point'].required = False
#         self.fields['route'].required = False

#     def clean_geojson_file(self):
#         """Validate and parse GeoJSON file"""
#         geojson_file = self.cleaned_data.get('geojson_file')

#         if geojson_file:
#             try:
#                 # Read file content
#                 file_content = geojson_file.read()

#                 # Try to decode as UTF-8
#                 try:
#                     geojson_data = json.loads(file_content.decode('utf-8'))
#                 except UnicodeDecodeError:
#                     raise forms.ValidationError("GeoJSON file must be UTF-8 encoded JSON.")

#                 # Validate GeoJSON structure
#                 if not isinstance(geojson_data, dict):
#                     raise forms.ValidationError("GeoJSON file must contain a valid GeoJSON object.")

#                 # Handle FeatureCollection
#                 if geojson_data.get('type') == 'FeatureCollection':
#                     features = geojson_data.get('features', [])
#                     if not features:
#                         raise forms.ValidationError("GeoJSON FeatureCollection is empty.")
#                     # Use first feature
#                     feature = features[0]
#                 # Handle Feature
#                 elif geojson_data.get('type') == 'Feature':
#                     feature = geojson_data
#                 # Handle direct Geometry
#                 elif geojson_data.get('type') in ['LineString', 'MultiLineString']:
#                     feature = {
#                         'type': 'Feature',
#                         'geometry': geojson_data,
#                         'properties': {}
#                     }
#                 else:
#                     raise forms.ValidationError(
#                         f"GeoJSON type '{geojson_data.get('type')}' not supported. "
#                         "Expected Feature, FeatureCollection, or LineString geometry."
#                     )

#                 # Validate geometry type
#                 geometry = feature.get('geometry', {})
#                 geometry_type = geometry.get('type')

#                 if geometry_type == 'LineString':
#                     coordinates = geometry.get('coordinates', [])
#                     if len(coordinates) < 2:
#                         raise forms.ValidationError("LineString must have at least 2 coordinates.")
#                 elif geometry_type == 'MultiLineString':
#                     # Use first LineString from MultiLineString
#                     coordinates = geometry.get('coordinates', [])
#                     if not coordinates or len(coordinates[0]) < 2:
#                         raise forms.ValidationError("MultiLineString must contain at least one LineString with 2+ coordinates.")
#                     coordinates = coordinates[0]
#                     # Update geometry to be LineString
#                     geometry = {
#                         'type': 'LineString',
#                         'coordinates': coordinates
#                     }
#                     feature['geometry'] = geometry
#                 else:
#                     raise forms.ValidationError(
#                         f"Geometry type '{geometry_type}' not supported. "
#                         "Expected LineString or MultiLineString."
#                     )

#                 # Store parsed data for use in save method
#                 self.geojson_data = {
#                     'geometry': geometry,
#                     'properties': feature.get('properties', {})
#                 }

#                 return geojson_file

#             except json.JSONDecodeError as e:
#                 raise forms.ValidationError(f"Invalid JSON format: {str(e)}")
#             except Exception as e:
#                 raise forms.ValidationError(f"Error parsing GeoJSON: {str(e)}")

#         return geojson_file

#     def save(self, commit=True):
#         """Override save to populate geographic fields from GeoJSON"""
#         instance = super().save(commit=False)

#         geojson_file = self.cleaned_data.get('geojson_file')

#         if geojson_file and hasattr(self, 'geojson_data'):
#             try:
#                 # Get coordinates directly from GeoJSON to ensure all points are preserved
#                 geojson_geometry = self.geojson_data['geometry']
#                 coordinates = geojson_geometry.get('coordinates', [])

#                 if len(coordinates) < 2:
#                     raise ValueError("LineString must have at least 2 coordinates")

#                 # Create LineString directly from all coordinates
#                 # GeoJSON coordinates are [longitude, latitude]
#                 # Convert to tuples for LineString: (longitude, latitude) = (x, y)
#                 line_coords = [(coord[0], coord[1]) for coord in coordinates]

#                 # Create LineString with all coordinates to preserve the actual route
#                 route_geometry = LineString(line_coords, srid=4326)

#                 # Assign the complete LineString to the route
#                 instance.route = route_geometry

#                 # Extract start and end points from the coordinates
#                 if len(coordinates) >= 2:
#                     # First point: coordinates[0] is [longitude, latitude]
#                     instance.start_point = Point(coordinates[0][0], coordinates[0][1], srid=4326)
#                     # Last point
#                     instance.end_point = Point(coordinates[-1][0], coordinates[-1][1], srid=4326)

#                 # Automatically calculate and set satellite image bounds from the route
#                 # Calculate bounds with a 1km buffer around the route
#                 default_bounds = instance.calculate_default_bounds(buffer_km=1.0)
#                 if default_bounds:
#                     instance.default_bounds = default_bounds
#                     logger.info(f"Calculated default bounds for pipeline {instance.name} from GeoJSON route")

#                 # Use the created geometry for length calculation
#                 geometry = route_geometry

#                 # Extract properties if present
#                 properties = self.geojson_data.get('properties', {})

#                 # Update fields from properties if not already set in form
#                 if properties.get('name') and not instance.name:
#                     instance.name = properties['name']
#                 if properties.get('description') and not instance.description:
#                     instance.description = properties['description']
#                 if properties.get('length_km') is not None:
#                     try:
#                         length = float(properties['length_km'])
#                         if length > 0:
#                             instance.length_km = length
#                         else:
#                             # Calculate length from geometry if not provided
#                             instance.length_km = self._calculate_length_km(geometry)
#                     except (ValueError, TypeError):
#                         instance.length_km = self._calculate_length_km(geometry)
#                 elif not instance.length_km or instance.length_km == 0:
#                     # Calculate length from geometry if not set
#                     instance.length_km = self._calculate_length_km(geometry)

#                 if properties.get('diameter_mm') is not None:
#                     try:
#                         diameter = float(properties['diameter_mm'])
#                         if diameter >= 10:  # Minimum validation
#                             instance.diameter_mm = diameter
#                     except (ValueError, TypeError):
#                         pass

#                 if properties.get('material') in dict(Pipeline._meta.get_field('material').choices):
#                     instance.material = properties['material']

#                 if properties.get('status') in dict(Pipeline._meta.get_field('status').choices):
#                     instance.status = properties['status']

#             except Exception as e:
#                 # If there's an error processing GeoJSON, allow save to continue
#                 # but don't populate geographic fields
#                 # The form will still save with manually entered data
#                 # Store error message for debugging
#                 logger.warning(f"Error processing GeoJSON file: {str(e)}")
#                 # Note: In a production environment, you might want to show this error
#                 # to the user via messages framework in the admin

#         if commit:
#             instance.save()
#         return instance

#     def _calculate_length_km(self, geometry):
#         """Calculate length in kilometers from geometry"""
#         if isinstance(geometry, LineString):
#             # Transform to a projection that uses meters (e.g., UTM)
#             # For Nigeria, we can use EPSG:32632 (UTM Zone 32N)
#             try:
#                 geometry.transform(32632)
#                 length_meters = geometry.length
#                 length_km = length_meters / 1000.0
#                 # Transform back to WGS84
#                 geometry.transform(4326)
#                 return round(length_km, 2)
#             except:
#                 # Fallback: approximate calculation using Haversine
#                 coords = list(geometry.coords)
#                 total_length = 0.0
#                 for i in range(len(coords) - 1):
#                     lat1, lon1 = coords[i][1], coords[i][0]
#                     lat2, lon2 = coords[i+1][1], coords[i+1][0]

#                     # Haversine formula (approximate)
#                     R = 6371  # Earth radius in km
#                     dlat = radians(lat2 - lat1)
#                     dlon = radians(lon2 - lon1)
#                     a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
#                     c = 2 * asin(sqrt(a))
#                     total_length += R * c

#                 return round(total_length, 2)
#         return 0.0
