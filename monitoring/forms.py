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
        widget=forms.FileInput(
            attrs={"accept": ".json,.geojson", "class": "fileinput"}
        ),
    )

    class Meta:
        model = Pipeline
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make geographic fields optional when GeoJSON is uploaded
        self.fields["start_point"].required = False
        self.fields["end_point"].required = False
        self.fields["route"].required = False

    def clean_geojson_file(self):
        """Validate and parse GeoJSON file"""
        geojson_file = self.cleaned_data.get("geojson_file")

        if geojson_file:
            try:
                # Read file content
                file_content = geojson_file.read()

                # Try to decode as UTF-8
                try:
                    geojson_data = json.loads(file_content.decode("utf-8"))
                except UnicodeDecodeError:
                    raise forms.ValidationError(
                        "GeoJSON file must be UTF-8 encoded JSON."
                    )

                # Validate GeoJSON structure
                if not isinstance(geojson_data, dict):
                    raise forms.ValidationError(
                        "GeoJSON file must contain a valid GeoJSON object."
                    )

                # Handle FeatureCollection
                if geojson_data.get("type") == "FeatureCollection":
                    features = geojson_data.get("features", [])
                    if not features:
                        raise forms.ValidationError(
                            "GeoJSON FeatureCollection is empty."
                        )
                    # Use first feature
                    feature = features[0]
                # Handle Feature
                elif geojson_data.get("type") == "Feature":
                    feature = geojson_data
                # Handle direct Geometry
                elif geojson_data.get("type") in ["LineString", "MultiLineString"]:
                    feature = {
                        "type": "Feature",
                        "geometry": geojson_data,
                        "properties": {},
                    }
                else:
                    raise forms.ValidationError(
                        f"GeoJSON type '{geojson_data.get('type')}' not supported. "
                        "Expected Feature, FeatureCollection, or LineString geometry."
                    )

                # Validate geometry type
                geometry = feature.get("geometry", {})
                geometry_type = geometry.get("type")

                if geometry_type == "LineString":
                    coordinates = geometry.get("coordinates", [])
                    if len(coordinates) < 2:
                        raise forms.ValidationError(
                            "LineString must have at least 2 coordinates."
                        )
                elif geometry_type == "MultiLineString":
                    # Use first LineString from MultiLineString
                    coordinates = geometry.get("coordinates", [])
                    if not coordinates or len(coordinates[0]) < 2:
                        raise forms.ValidationError(
                            "MultiLineString must contain at least one LineString with 2+ coordinates."
                        )
                    coordinates = coordinates[0]
                    # Update geometry to be LineString
                    geometry = {"type": "LineString", "coordinates": coordinates}
                    feature["geometry"] = geometry
                else:
                    raise forms.ValidationError(
                        f"Geometry type '{geometry_type}' not supported. "
                        "Expected LineString or MultiLineString."
                    )

                # Store parsed data for use in save method
                self.geojson_data = {
                    "geometry": geometry,
                    "properties": feature.get("properties", {}),
                }

                return geojson_file

            except json.JSONDecodeError as e:
                raise forms.ValidationError(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                raise forms.ValidationError(f"Error parsing GeoJSON: {str(e)}")

        return geojson_file

    def save(self, commit=True):
        """Override save to populate geographic fields from GeoJSON"""
        instance = super().save(commit=False)

        geojson_file = self.cleaned_data.get("geojson_file")

        if geojson_file and hasattr(self, "geojson_data"):
            try:
                # Get coordinates directly from GeoJSON to ensure all points are preserved
                geojson_geometry = self.geojson_data["geometry"]
                coordinates = geojson_geometry.get("coordinates", [])

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
                    instance.start_point = Point(
                        coordinates[0][0], coordinates[0][1], srid=4326
                    )
                    # Last point
                    instance.end_point = Point(
                        coordinates[-1][0], coordinates[-1][1], srid=4326
                    )

                # Use the created geometry for length calculation
                geometry = route_geometry

                # Extract properties if present
                properties = self.geojson_data.get("properties", {})

                # Update fields from properties if not already set in form
                if properties.get("name") and not instance.name:
                    instance.name = properties["name"]
                if properties.get("description") and not instance.description:
                    instance.description = properties["description"]
                if properties.get("length_km") is not None:
                    try:
                        length = float(properties["length_km"])
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

                if properties.get("diameter_mm") is not None:
                    try:
                        diameter = float(properties["diameter_mm"])
                        if diameter >= 10:  # Minimum validation
                            instance.diameter_mm = diameter
                    except (ValueError, TypeError):
                        pass

                if properties.get("material") in dict(
                    Pipeline._meta.get_field("material").choices
                ):
                    instance.material = properties["material"]

                if properties.get("status") in dict(
                    Pipeline._meta.get_field("status").choices
                ):
                    instance.status = properties["status"]

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
                    lat2, lon2 = coords[i + 1][1], coords[i + 1][0]

                    # Haversine formula (approximate)
                    R = 6371  # Earth radius in km
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = (
                        sin(dlat / 2) ** 2
                        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                    )
                    c = 2 * asin(sqrt(a))
                    total_length += R * c

                return round(total_length, 2)
        return 0.0
