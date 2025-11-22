# Deployment Guide

This guide covers deploying the Medical Impact Predictor to production environments.

## 🐳 Docker Deployment (Recommended)

### Create Dockerfile for Backend

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config/ ./config/
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

### Create Dockerfile for Frontend

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy source
COPY frontend/ .

# Build
RUN npm run build

# Serve with simple http server
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models:ro

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://backend:5000
```

Run with:
```bash
docker-compose up -d
```

## ☁️ Cloud Deployment Options

### Heroku

**Backend:**
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create medical-predictor-api
git push heroku main
```

**Frontend:**
```bash
cd frontend
heroku create medical-predictor-ui
heroku buildpacks:set mars/create-react-app
git push heroku main
```

### AWS (EC2)

1. **Launch EC2 instance** (t2.medium recommended)
2. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip nodejs npm
   ```
3. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd medical-impact-predictor
   pip3 install -r requirements.txt
   ```
4. **Run with systemd:**
   ```ini
   # /etc/systemd/system/medical-api.service
   [Unit]
   Description=Medical Impact Predictor API
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/medical-impact-predictor
   ExecStart=/usr/bin/python3 app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl enable medical-api
   sudo systemctl start medical-api
   ```

### Google Cloud Platform (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/medical-predictor

# Deploy
gcloud run deploy medical-predictor \
  --image gcr.io/PROJECT_ID/medical-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔒 Production Considerations

### Security

1. **Environment Variables:**
   ```bash
   # Use .env file (never commit!)
   FLASK_SECRET_KEY=your-secret-key
   DATABASE_URL=postgresql://...
   API_KEY=your-api-key
   ```

2. **HTTPS/SSL:**
   - Use Let's Encrypt for free SSL certificates
   - Configure nginx as reverse proxy

3. **Rate Limiting:**
   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(
       app,
       key_func=lambda: request.remote_addr,
       default_limits=["100 per hour"]
   )
   ```

### Performance

1. **Gunicorn (Production WSGI):**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Caching:**
   ```python
   from flask_caching import Cache
   
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   
   @cache.memoize(timeout=300)
   def predict_los(patient_data):
       # ... prediction logic
   ```

3. **Load Balancing:**
   - Use nginx or AWS ELB
   - Run multiple backend instances

### Monitoring

1. **Logging:**
   ```python
   import logging
   
   logging.basicConfig(
       filename='app.log',
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s'
   )
   ```

2. **Health Checks:**
   ```python
   @app.route('/health')
   def health_check():
       return jsonify({'status': 'healthy'}), 200
   ```

3. **Metrics:**
   - Use Prometheus + Grafana
   - Track prediction latency, error rates

## 📊 Database Integration (Optional)

For storing predictions:

```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://...'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    patient_data = db.Column(db.JSON)
    prediction_result = db.Column(db.JSON)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          python scripts/run_api_tests.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{secrets.HEROKU_API_KEY}}
          heroku_app_name: "medical-predictor"
          heroku_email: "your-email@example.com"
```

## 📝 Checklist Before Deployment

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] HTTPS/SSL enabled
- [ ] CORS properly configured
- [ ] Error handling tested
- [ ] Logging configured
- [ ] Health check endpoint working
- [ ] Database backups scheduled (if applicable)
- [ ] Monitoring/alerting setup
- [ ] Documentation updated

## 🆘 Troubleshooting

**Issue: High memory usage**
- Solution: Reduce model size or use model quantization

**Issue: Slow predictions**
- Solution: Implement caching, use faster model formats (ONNX)

**Issue: CORS errors in production**
- Solution: Update CORS origins in `app.py`

## 📧 Support

For deployment issues, contact: shiva.chandra@example.com
