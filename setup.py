#!/usr/bin/env python3
"""
Setup script for Pipeline Monitoring System Backend
"""

import os
import sys
import subprocess
import django
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = ['postgresql', 'redis-server', 'gdal-bin']
    missing = []
    
    for dep in dependencies:
        if not run_command(f"which {dep}", f"Checking {dep}"):
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please install the missing dependencies and run the setup again")
        sys.exit(1)

def setup_environment():
    """Set up environment variables"""
    env_file = Path('.env')
    if not env_file.exists():
        print("🔄 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# Django Settings
SECRET_KEY=django-insecure-change-this-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Settings
DB_NAME=pipeline_monitoring
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432

# Redis Settings
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# NASA API
NASA_API_KEY=your-nasa-api-key-here

# Email Settings
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@pipeline-monitoring.com
""")
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

def setup_database():
    """Set up PostgreSQL database"""
    print("🔄 Setting up database...")
    
    # Check if PostgreSQL is running
    if not run_command("pg_isready", "Checking PostgreSQL connection"):
        print("❌ PostgreSQL is not running. Please start PostgreSQL and run the setup again")
        sys.exit(1)
    
    # Create database if it doesn't exist
    run_command(
        "psql -U postgres -c 'CREATE DATABASE pipeline_monitoring;' || true",
        "Creating database"
    )
    
    # Enable PostGIS extension
    run_command(
        "psql -U postgres -d pipeline_monitoring -c 'CREATE EXTENSION IF NOT EXISTS postgis;'",
        "Enabling PostGIS extension"
    )
    
    print("✅ Database setup completed")

def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)

def run_migrations():
    """Run Django migrations"""
    if not run_command("python manage.py migrate", "Running database migrations"):
        print("❌ Failed to run migrations")
        sys.exit(1)

def create_superuser():
    """Create Django superuser"""
    print("🔄 Creating superuser...")
    try:
        from django.contrib.auth.models import User
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@pipeline-monitoring.com', 'admin123')
            print("✅ Superuser created (username: admin, password: admin123)")
        else:
            print("✅ Superuser already exists")
    except Exception as e:
        print(f"❌ Failed to create superuser: {e}")

def seed_sample_data():
    """Seed sample data"""
    if not run_command("python manage.py seed_data", "Seeding sample data"):
        print("❌ Failed to seed sample data")
        sys.exit(1)

def main():
    """Main setup function"""
    print("🚀 Setting up Pipeline Monitoring System Backend")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check system dependencies
    check_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Install Python dependencies
    install_dependencies()
    
    # Set up database
    setup_database()
    
    # Run migrations
    run_migrations()
    
    # Create superuser
    create_superuser()
    
    # Seed sample data
    seed_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your NASA API key")
    print("2. Start the development server: python manage.py runserver")
    print("3. Access the admin panel: http://localhost:8000/admin")
    print("4. View API documentation: http://localhost:8000/swagger/")
    print("\nDefault credentials:")
    print("Username: admin")
    print("Password: admin123")

if __name__ == "__main__":
    main()
