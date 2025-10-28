"""
File validation functions for satellite image uploads
"""
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
import os


# Allowed file extensions for TIFF format
ALLOWED_TIFF_EXTENSIONS = ['.tif', '.tiff', '.TIFF', '.TIF']

# Maximum file size in bytes (default: 100 MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Maximum file size in megabytes for display
MAX_FILE_SIZE_MB = 100


def validate_tiff_file(file):
    """
    Validate that the uploaded file is a TIFF format and within size limits.
    
    Args:
        file: The file object to validate
        
    Raises:
        ValidationError: If the file is not a TIFF format or exceeds size limits
    """
    # Check file extension
    file_extension = os.path.splitext(file.name)[1]
    if file_extension not in ALLOWED_TIFF_EXTENSIONS:
        allowed_extensions = ', '.join(ALLOWED_TIFF_EXTENSIONS)
        raise ValidationError(
            _(
                f'Invalid file format. Only TIFF files ({allowed_extensions}) are allowed.'
            )
        )
    
    # Check file size
    if file.size > MAX_FILE_SIZE_BYTES:
        file_size_mb = round(file.size / (1024 * 1024), 2)
        raise ValidationError(
            _(
                f'File size ({file_size_mb} MB) exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB.'
            )
        )
    
    # Check if file is too small (might be empty or corrupted)
    if file.size < 1024:  # Less than 1 KB
        raise ValidationError(
            _('File appears to be empty or corrupted.')
        )


def validate_file_size(file):
    """
    Validate only the file size without checking format.
    
    Args:
        file: The file object to validate
        
    Raises:
        ValidationError: If the file exceeds size limits
    """
    if file.size > MAX_FILE_SIZE_BYTES:
        file_size_mb = round(file.size / (1024 * 1024), 2)
        raise ValidationError(
            _(
                f'File size ({file_size_mb} MB) exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB.'
            )
        )


def validate_thumbnail_file(file):
    """
    Validate thumbnail file (typically smaller images).
    Allows JPG, PNG, and TIFF formats with smaller size limits.
    
    Args:
        file: The file object to validate
        
    Raises:
        ValidationError: If the file format is not supported or exceeds size limits
    """
    # Allowed extensions for thumbnails
    ALLOWED_THUMBNAIL_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    MAX_THUMBNAIL_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
    
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension not in ALLOWED_THUMBNAIL_EXTENSIONS:
        allowed_extensions = ', '.join(ALLOWED_THUMBNAIL_EXTENSIONS)
        raise ValidationError(
            _(
                f'Invalid thumbnail format. Only {allowed_extensions} files are allowed.'
            )
        )
    
    if file.size > MAX_THUMBNAIL_SIZE_BYTES:
        file_size_mb = round(file.size / (1024 * 1024), 2)
        raise ValidationError(
            _(
                f'Thumbnail size ({file_size_mb} MB) exceeds the maximum allowed size of 5 MB.'
            )
        )


def get_allowed_extensions():
    """
    Return the list of allowed file extensions for TIFF files.
    """
    return ALLOWED_TIFF_EXTENSIONS


def get_max_file_size():
    """
    Return the maximum file size in bytes.
    """
    return MAX_FILE_SIZE_BYTES


def get_max_file_size_mb():
    """
    Return the maximum file size in megabytes.
    """
    return MAX_FILE_SIZE_MB

