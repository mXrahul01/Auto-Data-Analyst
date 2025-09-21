# Deployment Guide 🚀

[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.21+-326ce5.svg)](https://kubernetes.io)
[![AWS](https://img.shields.io/badge/AWS-Compatible-orange.svg)](https://aws.amazon.com)
[![GCP](https://img.shields.io/badge/GCP-Compatible-blue.svg)](https://cloud.google.com)

## 🌟 Overview

This guide covers comprehensive deployment strategies for the Auto Data Analyst platform across multiple environments, from local development to enterprise-grade production deployments. The platform is designed for cloud-native deployment with support for containerization, orchestration, and multi-cloud environments.

### 🎯 Deployment Options

- **🔥 Quick Deploy** - Render, Railway, Heroku (Recommended for demos)
- **🏢 Production** - AWS, GCP, Azure with Kubernetes
- **🐳 Container** - Docker and Docker Compose
- **☁️ Cloud-Native** - Serverless and managed services
- **🏠 On-Premise** - Private cloud and bare metal

---

## ⚡ Quick Deployment (5 Minutes)

### 🟢 Render (Recommended)

**Perfect for demos, portfolios, and small-scale production**

#### Step 1: Prepare Repository
```bash
# Ensure these files exist in your repo
├── requirements.txt
├── main.py
├── .env.example
├── .gitignore
└── backend/
```

#### Step 2: Deploy to Render
1. **Connect Repository**
   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select `mXrahul01/Auto-Data-Analyst`

2. **Configure Build Settings**
   ```
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. **Set Environment Variables**
   ```bash
   SECRET_KEY=your-super-secure-secret-key-here
   ENVIRONMENT=production
   DATABASE_URL=sqlite:///./app.db
   DEBUG=false
   LOG_LEVEL=INFO
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Access your API at: `https://your-app-name.onrender.com`

#### Step 3: Verify Deployment
```bash
curl https://your-app-name.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "services": {
    "database": "healthy",
    "ml_pipeline": "healthy"
  }
}
```

### 🚄 Railway

**Alternative quick deployment platform**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway add database
railway deploy
```

### 📊 Heroku

**Classic PaaS deployment**

```bash
# Install Heroku CLI and login
heroku login

# Create application
heroku create auto-data-analyst

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set ENVIRONMENT=production

# Deploy
git push heroku main

# Scale dynos
heroku ps:scale web=1
```

---

## 🐳 Docker Deployment

### Single Container Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads temp models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Build and Run
```bash
# Build image
docker build -t auto-data-analyst .

# Run container
docker run -d \
  --name auto-analyst \
  -p 8000:8000 \
  -e SECRET_KEY=your-secret-key \
  -e ENVIRONMENT=production \
  -v $(pwd)/data:/app/data \
  auto-data-analyst

# Check logs
docker logs auto-analyst

# Check health
curl http://localhost:8000/health
```

### Docker Compose (Full Stack)

#### docker-compose.yml
```yaml
version: '3.8'

services:
  # Main application
  app:
    build: .
    container_name: auto-analyst-app
    ports:
      - "8000:8000"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/autoanalyst
      - REDIS_URL=redis://redis:6379/0
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - db
      - redis
      - mlflow
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Database
  db:
    image: postgres:15
    container_name: auto-analyst-db
    environment:
      - POSTGRES_DB=autoanalyst
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis for caching and job queue
  redis:
    image: redis:7-alpine
    container_name: auto-analyst-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # MLflow for experiment tracking
  mlflow:
    image: python:3.11-slim
    container_name: auto-analyst-mlflow
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_data:/mlflow
    depends_on:
      - db
    restart: unless-stopped
    command: >
      sh -c "
        pip install mlflow psycopg2-binary &&
        mlflow server 
          --backend-store-uri postgresql://postgres:password@db:5432/mlflow
          --default-artifact-root /mlflow/artifacts
          --host 0.0.0.0
          --port 5000
      "

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: auto-analyst-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus
    container_name: auto-analyst-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  # Grafana dashboards
  grafana:
    image: grafana/grafana
    container_name: auto-analyst-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

#### Deploy Full Stack
```bash
# Create environment file
cp .env.example .env

# Edit environment variables
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3
```

#### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # File upload handling
        location /api/v1/datasets/upload {
            client_max_body_size 20G;
            proxy_pass http://app;
            proxy_request_buffering off;
            proxy_set_header Host $host;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://app;
        }
    }
}
```

---

## ☁️ Cloud Platform Deployments

### 🔶 AWS Deployment

#### ECS Fargate with RDS

**1. Infrastructure Setup**
```bash
# Create VPC and subnets
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 create-subnet --vpc-id vpc-xxx --cidr-block 10.0.1.0/24
aws ec2 create-subnet --vpc-id vpc-xxx --cidr-block 10.0.2.0/24

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier auto-analyst-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --master-username autoanalyst \
  --master-user-password ${DB_PASSWORD} \
  --allocated-storage 100 \
  --vpc-security-group-ids sg-xxx
```

**2. ECS Task Definition**
```json
{
  "family": "auto-data-analyst",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/autoAnalystTaskRole",
  "containerDefinitions": [
    {
      "name": "auto-analyst-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/auto-data-analyst:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "CLOUD_PROVIDER", 
          "value": "aws"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:auto-analyst/secret-key"
        },
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:auto-analyst/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/auto-data-analyst",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

**3. Application Load Balancer**
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name auto-analyst-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
  --name auto-analyst-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx \
  --health-check-path /health
```

#### EKS (Kubernetes) Deployment

**1. Cluster Setup**
```bash
# Create EKS cluster
eksctl create cluster \
  --name auto-analyst-cluster \
  --version 1.21 \
  --region us-east-1 \
  --nodegroup-name workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed
```

**2. Kubernetes Manifests**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: auto-analyst

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: auto-analyst-config
  namespace: auto-analyst
data:
  ENVIRONMENT: "production"
  CLOUD_PROVIDER: "aws"
  LOG_LEVEL: "INFO"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: auto-analyst-secrets
  namespace: auto-analyst
type: Opaque
stringData:
  SECRET_KEY: "your-secret-key"
  DATABASE_URL: "postgresql://user:pass@rds-endpoint:5432/autoanalyst"

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auto-analyst-app
  namespace: auto-analyst
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auto-analyst-app
  template:
    metadata:
      labels:
        app: auto-analyst-app
    spec:
      containers:
      - name: app
        image: your-account.dkr.ecr.region.amazonaws.com/auto-data-analyst:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: auto-analyst-config
        - secretRef:
            name: auto-analyst-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: auto-analyst-service
  namespace: auto-analyst
spec:
  selector:
    app: auto-analyst-app
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: auto-analyst-ingress
  namespace: auto-analyst
  annotations:
    kubernetes.io/ingress.class: "alb"
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-redirect: "443"
spec:
  rules:
  - host: api.autoanalyst.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: auto-analyst-service
            port:
              number: 80
```

**3. Deploy to EKS**
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n auto-analyst
kubectl get services -n auto-analyst
kubectl get ingress -n auto-analyst

# Scale deployment
kubectl scale deployment auto-analyst-app --replicas=5 -n auto-analyst
```

### 🔵 Google Cloud Platform (GCP)

#### Cloud Run Deployment

**1. Build and Push to Container Registry**
```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Build and tag image
docker build -t gcr.io/PROJECT-ID/auto-data-analyst .

# Push to Container Registry
docker push gcr.io/PROJECT-ID/auto-data-analyst
```

**2. Deploy to Cloud Run**
```bash
# Deploy service
gcloud run deploy auto-data-analyst \
  --image gcr.io/PROJECT-ID/auto-data-analyst \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 100 \
  --set-env-vars ENVIRONMENT=production,CLOUD_PROVIDER=gcp \
  --set-secrets SECRET_KEY=auto-analyst-secret:latest,DATABASE_URL=database-url:latest
```

#### GKE Deployment

**1. Create GKE Cluster**
```bash
# Create cluster
gcloud container clusters create auto-analyst-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --machine-type n1-standard-2 \
  --enable-autorepair \
  --enable-autoupgrade
```

**2. Deploy Application**
```bash
# Get cluster credentials
gcloud container clusters get-credentials auto-analyst-cluster --zone us-central1-a

# Deploy using kubectl (same manifests as EKS)
kubectl apply -f k8s/
```

### 🟦 Microsoft Azure

#### Container Instances Deployment
```bash
# Create resource group
az group create --name auto-analyst-rg --location eastus

# Create container instance
az container create \
  --resource-group auto-analyst-rg \
  --name auto-analyst-app \
  --image your-registry.azurecr.io/auto-data-analyst:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production CLOUD_PROVIDER=azure \
  --secure-environment-variables SECRET_KEY=your-secret-key DATABASE_URL=your-db-url \
  --dns-name-label auto-analyst-demo
```

#### AKS Deployment
```bash
# Create AKS cluster
az aks create \
  --resource-group auto-analyst-rg \
  --name auto-analyst-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group auto-analyst-rg --name auto-analyst-cluster

# Deploy application
kubectl apply -f k8s/
```

---

## 📊 Database Deployment Options

### SQLite (Development/Small Scale)
```python
# Configuration for SQLite
DATABASE_URL = "sqlite:///./autoanalyst.db"
```

### PostgreSQL (Recommended Production)

#### Local PostgreSQL
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createuser --interactive autoanalyst
sudo -u postgres createdb autoanalyst -O autoanalyst
sudo -u postgres psql -c "ALTER USER autoanalyst PASSWORD 'password';"
```

#### Managed PostgreSQL Services

**AWS RDS:**
```bash
aws rds create-db-instance \
  --db-instance-identifier auto-analyst-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 13.7 \
  --master-username autoanalyst \
  --master-user-password ${DB_PASSWORD} \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-xxx \
  --backup-retention-period 7 \
  --multi-az
```

**Google Cloud SQL:**
```bash
gcloud sql instances create auto-analyst-db \
  --database-version POSTGRES_13 \
  --tier db-f1-micro \
  --region us-central1 \
  --storage-size 100GB \
  --storage-type SSD \
  --backup-start-time 03:00
```

**Azure Database for PostgreSQL:**
```bash
az postgres server create \
  --resource-group auto-analyst-rg \
  --name auto-analyst-db \
  --location eastus \
  --admin-user autoanalyst \
  --admin-password ${DB_PASSWORD} \
  --sku-name B_Gen5_1 \
  --version 11
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

#### .github/workflows/deploy.yml
```yaml
name: Deploy Auto Data Analyst

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
        SECRET_KEY: test-secret-key
        ENVIRONMENT: testing
      run: |
        pytest --cov=backend --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        echo "Deploying to staging..."
        # Add your staging deployment commands here

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Deploy to production
      run: |
        # Deploy to production environment
        echo "Deploying to production..."
        # Add your production deployment commands here
```

### GitLab CI/CD

#### .gitlab-ci.yml
```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.11
  services:
    - postgres:13
  variables:
    POSTGRES_DB: test
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test
    SECRET_KEY: test-secret-key
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-test.txt
  script:
    - pytest --cov=backend --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy-staging:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Deploying to staging..."
    # Add your staging deployment commands
  environment:
    name: staging
    url: https://staging.autoanalyst.com
  only:
    - main

deploy-production:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Deploying to production..."
    # Add your production deployment commands
  environment:
    name: production
    url: https://api.autoanalyst.com
  only:
    - main
  when: manual
```

---

## 📊 Monitoring & Observability

### Application Monitoring

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'auto-analyst'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Auto Data Analyst Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Times",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active ML Analyses",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_ml_analyses",
            "legendFormat": "Active Analyses"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

#### ELK Stack Setup
```yaml
# elasticsearch.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

#### Logstash Configuration
```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "auto-analyst" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => ["error"]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "auto-analyst-%{+YYYY.MM.dd}"
  }
  
  if "error" in [tags] {
    email {
      to => ["admin@autoanalyst.com"]
      subject => "Auto Analyst Error Alert"
      body => "Error occurred: %{message}"
    }
  }
}
```

---

## 🔒 Security & Best Practices

### Security Configuration

#### Environment Variables Security
```bash
# Use secret management services
export SECRET_KEY=$(aws secretsmanager get-secret-value --secret-id auto-analyst/secret-key --query SecretString --output text)

# Rotate secrets regularly
aws secretsmanager rotate-secret --secret-id auto-analyst/secret-key
```

#### SSL/TLS Configuration
```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name api.autoanalyst.com;

    ssl_certificate /etc/ssl/certs/autoanalyst.crt;
    ssl_certificate_key /etc/ssl/private/autoanalyst.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Network Security
```yaml
# kubernetes-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: auto-analyst-network-policy
  namespace: auto-analyst
spec:
  podSelector:
    matchLabels:
      app: auto-analyst-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
```

---

## 🚀 Performance Optimization

### Horizontal Pod Autoscaling
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: auto-analyst-hpa
  namespace: auto-analyst
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: auto-analyst-app
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaling
```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: auto-analyst-vpa
  namespace: auto-analyst
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: auto-analyst-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: app
      maxAllowed:
        cpu: "2"
        memory: "4Gi"
      minAllowed:
        cpu: "100m"
        memory: "128Mi"
```

---

## 🛠️ Maintenance & Operations

### Backup Strategy

#### Database Backups
```bash
#!/bin/bash
# backup-db.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="autoanalyst"

# Create backup
pg_dump $DATABASE_URL > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://auto-analyst-backups/

# Cleanup old local backups (keep 7 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

#### Application State Backup
```bash
#!/bin/bash
# backup-app-state.sh
DATE=$(date +%Y%m%d_%H%M%S)

# Backup uploaded files
tar -czf /backups/uploads_$DATE.tar.gz /app/uploads/

# Backup models
tar -czf /backups/models_$DATE.tar.gz /app/models/

# Backup configuration
cp /app/.env /backups/env_$DATE.backup

# Upload to cloud storage
aws s3 cp /backups/uploads_$DATE.tar.gz s3://auto-analyst-backups/
aws s3 cp /backups/models_$DATE.tar.gz s3://auto-analyst-backups/
```

### Health Checks & Monitoring

#### Comprehensive Health Check
```python
# health_check.py
import asyncio
import aiohttp
import psycopg2
import redis
from datetime import datetime

async def health_check():
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "checks": {}
    }
    
    # Database check
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        status["checks"]["database"] = "healthy"
    except Exception as e:
        status["checks"]["database"] = f"unhealthy: {e}"
        status["status"] = "degraded"
    
    # Redis check
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        status["checks"]["redis"] = "healthy"
    except Exception as e:
        status["checks"]["redis"] = f"unhealthy: {e}"
        status["status"] = "degraded"
    
    # ML Pipeline check
    try:
        # Test ML pipeline availability
        from backend.ml.auto_pipeline import AutoMLPipeline
        pipeline = AutoMLPipeline()
        status["checks"]["ml_pipeline"] = "healthy"
    except Exception as e:
        status["checks"]["ml_pipeline"] = f"unhealthy: {e}"
        status["status"] = "degraded"
    
    return status
```

### Deployment Rollback

#### Kubernetes Rollback
```bash
# Check deployment history
kubectl rollout history deployment/auto-analyst-app -n auto-analyst

# Rollback to previous version
kubectl rollout undo deployment/auto-analyst-app -n auto-analyst

# Rollback to specific revision
kubectl rollout undo deployment/auto-analyst-app --to-revision=2 -n auto-analyst

# Check rollback status
kubectl rollout status deployment/auto-analyst-app -n auto-analyst
```

#### Docker Compose Rollback
```bash
# Tag current version as backup
docker tag auto-data-analyst:latest auto-data-analyst:backup

# Pull previous version
docker pull auto-data-analyst:previous

# Stop current services
docker-compose down

# Update image tag in docker-compose.yml
sed -i 's/auto-data-analyst:latest/auto-data-analyst:previous/g' docker-compose.yml

# Restart services
docker-compose up -d
```

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] **Code Quality**
  - [ ] All tests passing
  - [ ] Code coverage > 90%
  - [ ] Security scan completed
  - [ ] Performance benchmarks met

- [ ] **Configuration**
  - [ ] Environment variables configured
  - [ ] Secrets properly managed
  - [ ] Database migrations ready
  - [ ] SSL certificates obtained

- [ ] **Dependencies**
  - [ ] requirements.txt updated
  - [ ] Docker images built and tested
  - [ ] External services configured
  - [ ] Monitoring tools setup

### Post-Deployment
- [ ] **Verification**
  - [ ] Health checks passing
  - [ ] API endpoints responding
  - [ ] Database connectivity verified
  - [ ] ML pipeline functional

- [ ] **Monitoring**
  - [ ] Metrics collection active
  - [ ] Log aggregation working
  - [ ] Alerts configured
  - [ ] Performance monitoring enabled

- [ ] **Documentation**
  - [ ] Deployment notes updated
  - [ ] Runbooks created
  - [ ] Team notified
  - [ ] Changelog updated

---

## 🆘 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs auto-analyst-app

# Check resource usage
docker stats auto-analyst-app

# Inspect container
docker inspect auto-analyst-app

# Debug with interactive shell
docker run -it --entrypoint /bin/bash auto-data-analyst
```

#### Database Connection Issues
```bash
# Test database connectivity
docker run --rm postgres:13 psql $DATABASE_URL -c "SELECT version();"

# Check network connectivity
docker run --rm alpine:latest nc -zv db-host 5432

# Verify credentials
docker run --rm postgres:13 psql $DATABASE_URL -c "\l"
```

#### High Memory Usage
```bash
# Check memory usage by process
docker exec auto-analyst-app ps aux --sort=-%mem

# Monitor memory usage over time  
docker exec auto-analyst-app free -h

# Check for memory leaks
docker exec auto-analyst-app cat /proc/meminfo
```

#### SSL/TLS Issues
```bash
# Test SSL certificate
openssl s_client -connect api.autoanalyst.com:443 -servername api.autoanalyst.com

# Check certificate expiration
echo | openssl s_client -connect api.autoanalyst.com:443 2>/dev/null | openssl x509 -noout -dates

# Verify certificate chain
curl -vI https://api.autoanalyst.com/health
```

### Performance Optimization

#### Database Optimization
```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Analyze table statistics
ANALYZE;

-- Reindex tables
REINDEX DATABASE autoanalyst;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### Application Optimization
```python
# Profile application performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your application code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(20)
```

---

## 📞 Support & Resources

### Getting Help

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/mXrahul01/Auto-Data-Analyst/issues)
- **💬 Community**: [Discord Server](https://discord.gg/autoanalyst)
- **📧 Email**: support@autoanalyst.com
- **📚 Documentation**: [docs.autoanalyst.com](https://docs.autoanalyst.com)

### Additional Resources

- **🎥 Video Tutorials**: [YouTube Channel](https://youtube.com/@autoanalyst)
- **📝 Blog Posts**: [blog.autoanalyst.com](https://blog.autoanalyst.com)
- **🏗️ Architecture Guides**: [architecture.autoanalyst.com](https://architecture.autoanalyst.com)
- **🔧 Best Practices**: [bestpractices.autoanalyst.com](https://bestpractices.autoanalyst.com)

### Enterprise Support

For enterprise deployments and support:
- **Enterprise Sales**: enterprise@autoanalyst.com
- **Professional Services**: services@autoanalyst.com
- **Training**: training@autoanalyst.com

---

**🚀 Ready to deploy your Auto Data Analyst platform! Choose your deployment strategy and follow the step-by-step guides above.**

*For the latest deployment guides and updates, visit our [Deployment Documentation](https://docs.autoanalyst.com/deployment)*
