# Deployment Guide

This guide covers deploying the Pipeline Monitoring System backend to production.

## Prerequisites

- Docker and Docker Compose
- Domain name (for production)
- SSL certificates (for HTTPS)
- NASA API key
- PostgreSQL database (if not using Docker)
- Redis instance (if not using Docker)

## Environment Setup

### 1. Environment Variables

Create a `.env` file with production values:

```bash
# Django Settings
SECRET_KEY=your-production-secret-key
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database Settings
DB_NAME=pipeline_monitoring
DB_USER=postgres
DB_PASSWORD=your-secure-password
DB_HOST=db
DB_PORT=5432

# Redis Settings
REDIS_URL=redis://redis:6379/1
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# NASA API
NASA_API_KEY=your-nasa-api-key

# Email Settings
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@your-domain.com

# Security
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
```

### 2. SSL Certificates

For production, you'll need SSL certificates. You can use Let's Encrypt:

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com -d www.your-domain.com

# Copy certificates to ssl directory
sudo mkdir -p ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
sudo chown -R $USER:$USER ssl/
```

## Docker Deployment

### 1. Development Deployment

```bash
# Clone repository
git clone <repository-url>
cd pipeline-monitoring/backend

# Set up environment
cp env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Seed sample data
docker-compose exec web python manage.py seed_data
```

### 2. Production Deployment

```bash
# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate

# Collect static files
docker-compose -f docker-compose.prod.yml exec web python manage.py collectstatic --noinput

# Create superuser
docker-compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
```

## Manual Deployment

### 1. Server Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3.11 python3.11-venv python3-pip postgresql postgresql-contrib postgis redis-server nginx

# Install GDAL
sudo apt-get install -y gdal-bin libgdal-dev

# Create application user
sudo useradd -m -s /bin/bash pipeline
sudo usermod -aG sudo pipeline
```

### 2. Database Setup

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE pipeline_monitoring;
CREATE USER pipeline_user WITH PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE pipeline_monitoring TO pipeline_user;
\c pipeline_monitoring
CREATE EXTENSION postgis;
\q
```

### 3. Application Setup

```bash
# Switch to application user
sudo su - pipeline

# Clone repository
git clone <repository-url>
cd pipeline-monitoring/backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with production values

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput
```

### 4. Systemd Services

Create systemd service files for the application:

```bash
# Create service file
sudo nano /etc/systemd/system/pipeline-monitoring.service
```

```ini
[Unit]
Description=Pipeline Monitoring System
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=pipeline
Group=pipeline
WorkingDirectory=/home/pipeline/pipeline-monitoring/backend
Environment=PATH=/home/pipeline/pipeline-monitoring/backend/venv/bin
ExecStart=/home/pipeline/pipeline-monitoring/backend/venv/bin/gunicorn --bind 0.0.0.0:8000 --workers 3 pipeline_monitoring.wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Create Celery service file
sudo nano /etc/systemd/system/pipeline-monitoring-celery.service
```

```ini
[Unit]
Description=Pipeline Monitoring Celery Worker
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=pipeline
Group=pipeline
WorkingDirectory=/home/pipeline/pipeline-monitoring/backend
Environment=PATH=/home/pipeline/pipeline-monitoring/backend/venv/bin
ExecStart=/home/pipeline/pipeline-monitoring/backend/venv/bin/celery -A pipeline_monitoring worker -l info
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Create Celery Beat service file
sudo nano /etc/systemd/system/pipeline-monitoring-celery-beat.service
```

```ini
[Unit]
Description=Pipeline Monitoring Celery Beat
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=pipeline
Group=pipeline
WorkingDirectory=/home/pipeline/pipeline-monitoring/backend
Environment=PATH=/home/pipeline/pipeline-monitoring/backend/venv/bin
ExecStart=/home/pipeline/pipeline-monitoring/backend/venv/bin/celery -A pipeline_monitoring beat -l info
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

### 5. Start Services

```bash
# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable pipeline-monitoring
sudo systemctl enable pipeline-monitoring-celery
sudo systemctl enable pipeline-monitoring-celery-beat
sudo systemctl start pipeline-monitoring
sudo systemctl start pipeline-monitoring-celery
sudo systemctl start pipeline-monitoring-celery-beat
```

### 6. Nginx Configuration

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/pipeline-monitoring
```

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /home/pipeline/pipeline-monitoring/backend/staticfiles/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /home/pipeline/pipeline-monitoring/backend/media/;
        expires 1y;
        add_header Cache-Control "public";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/pipeline-monitoring /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring and Maintenance

### 1. Log Monitoring

```bash
# View application logs
sudo journalctl -u pipeline-monitoring -f

# View Celery logs
sudo journalctl -u pipeline-monitoring-celery -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. Database Maintenance

```bash
# Backup database
pg_dump -h localhost -U pipeline_user pipeline_monitoring > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
psql -h localhost -U pipeline_user pipeline_monitoring < backup_file.sql
```

### 3. Application Updates

```bash
# Pull latest changes
cd /home/pipeline/pipeline-monitoring/backend
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Install new dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Restart services
sudo systemctl restart pipeline-monitoring
sudo systemctl restart pipeline-monitoring-celery
sudo systemctl restart pipeline-monitoring-celery-beat
```

### 4. Health Checks

```bash
# Check service status
sudo systemctl status pipeline-monitoring
sudo systemctl status pipeline-monitoring-celery
sudo systemctl status pipeline-monitoring-celery-beat

# Check database connection
python manage.py dbshell

# Check Redis connection
redis-cli ping

# Check API health
curl -f http://localhost:8000/api/
```

## Security Considerations

### 1. Firewall Configuration

```bash
# Configure UFW
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. SSL/TLS Configuration

- Use strong SSL/TLS configurations
- Enable HSTS headers
- Use secure cipher suites
- Regular certificate renewal

### 3. Application Security

- Keep dependencies updated
- Use strong secret keys
- Enable security headers
- Implement rate limiting
- Regular security audits

## Scaling Considerations

### 1. Horizontal Scaling

- Multiple Celery workers
- Load balancer for web servers
- Database read replicas
- Redis clustering

### 2. Performance Optimization

- Database query optimization
- Caching strategies
- CDN for static files
- Image optimization

### 3. Monitoring

- Application performance monitoring
- Database monitoring
- Server resource monitoring
- Log aggregation

## Troubleshooting

### Common Issues

1. **Database Connection Errors**

   - Check PostgreSQL service status
   - Verify database credentials
   - Check network connectivity

2. **Celery Worker Issues**

   - Check Redis connection
   - Verify task queue status
   - Check worker logs

3. **Static File Issues**

   - Check file permissions
   - Verify Nginx configuration
   - Check collectstatic output

4. **SSL Certificate Issues**
   - Verify certificate validity
   - Check certificate paths
   - Renew expired certificates

### Support

For additional support:

- Check application logs
- Review system logs
- Contact system administrator
- Create GitHub issue

## Backup and Recovery

### 1. Database Backups

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/home/pipeline/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U pipeline_user pipeline_monitoring > $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

### 2. Media Files Backup

```bash
# Backup media files
rsync -av /home/pipeline/pipeline-monitoring/backend/media/ /backup/media/
```

### 3. Configuration Backup

```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env nginx.conf
```

This deployment guide provides comprehensive instructions for deploying the Pipeline Monitoring System backend in both development and production environments.
