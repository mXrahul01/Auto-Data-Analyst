# API Documentation 📚

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0+-blue.svg)](https://swagger.io/specification/)
[![REST](https://img.shields.io/badge/REST-API-orange.svg)](https://restfulapi.net/)

## 🌟 Overview

The Auto Data Analyst API provides a comprehensive RESTful interface for zero-code AI-powered data analysis. Built with FastAPI, it offers high-performance, async endpoints for dataset management, machine learning pipeline orchestration, and real-time insights generation.

### 🚀 API Features

- **RESTful Architecture** - Clean, predictable endpoints following REST principles
- **Async/Await Support** - High-performance concurrent request handling
- **OpenAPI/Swagger** - Auto-generated interactive documentation
- **Real-time Updates** - WebSocket support for live progress tracking
- **Authentication & Authorization** - JWT-based security with role-based access
- **Comprehensive Validation** - Pydantic models for request/response validation
- **Error Handling** - Standardized error responses with detailed messages
- **Rate Limiting** - Built-in request throttling and abuse prevention

---

## 🔗 Base URL & Versioning

### Production
```
https://auto-data-analyst-backend.onrender.com/api/v1
```

### Development
```
http://localhost:8000/api/v1
```

### Interactive Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

---

## 🔐 Authentication

### JWT Token-Based Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Authentication Endpoints

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 604800,
  "refresh_token": "refresh_token_here"
}
```

#### Token Refresh
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "string"
}
```

#### Logout
```http
POST /api/v1/auth/logout
Authorization: Bearer <token>
```

#### User Profile
```http
GET /api/v1/auth/me
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": 1,
  "username": "johndoe",
  "email": "john@example.com",
  "is_active": true,
  "is_verified": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

## 📊 Dataset Management

### Upload Dataset

Upload and validate datasets with support for multiple formats.

```http
POST /api/v1/datasets/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

{
  "file": <binary_file>,
  "name": "string (optional)",
  "description": "string (optional)",
  "tags": "comma,separated,tags (optional)"
}
```

**Supported Formats:**
- CSV (`.csv`, `.tsv`)
- Excel (`.xlsx`, `.xls`)  
- JSON (`.json`)
- Parquet (`.parquet`)
- Feather (`.feather`)

**Response:**
```json
{
  "id": 123,
  "name": "sales_data.csv",
  "original_filename": "sales_data.csv",
  "status": "processing",
  "file_size": 2048576,
  "content_type": "text/csv",
  "upload_time": "2025-01-01T10:30:00Z",
  "tags": ["sales", "monthly", "revenue"],
  "message": "Dataset uploaded successfully. Processing in background."
}
```

### List Datasets

Retrieve user's datasets with pagination and filtering.

```http
GET /api/v1/datasets?page=1&size=10&status=processed&search=sales
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `size` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (`processing`, `processed`, `failed`)
- `search` (string): Search in dataset names and descriptions
- `tags` (string): Filter by tags (comma-separated)

**Response:**
```json
{
  "items": [
    {
      "id": 123,
      "name": "sales_data.csv",
      "status": "processed",
      "file_size": 2048576,
      "num_rows": 10000,
      "num_columns": 15,
      "upload_time": "2025-01-01T10:30:00Z",
      "tags": ["sales", "monthly"]
    }
  ],
  "total": 1,
  "page": 1,
  "pages": 1,
  "size": 20
}
```

### Get Dataset Details

Retrieve detailed information about a specific dataset.

```http
GET /api/v1/datasets/{dataset_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": 123,
  "name": "sales_data.csv",
  "original_filename": "sales_data.csv",
  "status": "processed",
  "file_size": 2048576,
  "num_rows": 10000,
  "num_columns": 15,
  "column_names": ["date", "product", "revenue", "quantity"],
  "column_types": {
    "date": "datetime",
    "product": "categorical",
    "revenue": "numeric",
    "quantity": "numeric"
  },
  "data_quality_score": 0.92,
  "missing_values": 45,
  "duplicate_rows": 12,
  "upload_time": "2025-01-01T10:30:00Z",
  "processing_time": 15.6,
  "tags": ["sales", "monthly"],
  "metadata": {
    "encoding": "utf-8",
    "separator": ",",
    "file_format": "csv"
  }
}
```

### Dataset Preview

Get a preview of the dataset (first few rows).

```http
GET /api/v1/datasets/{dataset_id}/preview?rows=5
Authorization: Bearer <token>
```

**Response:**
```json
{
  "columns": ["date", "product", "revenue", "quantity"],
  "data": [
    ["2025-01-01", "Product A", 1500.00, 10],
    ["2025-01-02", "Product B", 2300.50, 15],
    ["2025-01-03", "Product A", 1800.25, 12]
  ],
  "total_rows": 10000,
  "preview_rows": 3,
  "data_types": {
    "date": "datetime",
    "product": "categorical",
    "revenue": "numeric",
    "quantity": "numeric"
  }
}
```

### Delete Dataset

Delete a dataset and all associated data.

```http
DELETE /api/v1/datasets/{dataset_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Dataset deleted successfully",
  "deleted_at": "2025-01-01T12:00:00Z"
}
```

---

## 🤖 ML Analysis

### Create Analysis

Start a new machine learning analysis on a dataset.

```http
POST /api/v1/analyses
Authorization: Bearer <token>
Content-Type: application/json

{
  "dataset_id": 123,
  "target_column": "revenue",
  "task_type": "regression",
  "execution_mode": "local_cpu",
  "algorithms": ["xgboost", "catboost", "random_forest"],
  "config": {
    "max_models": 10,
    "max_time": 1800,
    "cross_validation_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "enable_feature_engineering": true,
    "enable_hyperparameter_tuning": true
  }
}
```

**Task Types:**
- `classification` - Binary or multi-class classification
- `regression` - Numeric target prediction
- `time_series` - Time series forecasting
- `clustering` - Unsupervised clustering
- `anomaly_detection` - Outlier detection
- `text_analysis` - NLP and text mining
- `recommendation` - Recommendation systems
- `deep_learning` - Neural network models

**Execution Modes:**
- `local_cpu` - Local CPU processing
- `local_gpu` - Local GPU processing (if available)
- `remote_kaggle` - Kaggle Kernels
- `remote_colab` - Google Colab
- `cloud_aws` - AWS SageMaker
- `cloud_gcp` - Google Cloud Vertex AI
- `cloud_azure` - Azure ML

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "status": "created",
  "message": "Analysis created successfully. Starting execution.",
  "created_at": "2025-01-01T11:00:00Z",
  "estimated_completion": "2025-01-01T11:30:00Z"
}
```

### Get Analysis Status

Check the status and progress of a running analysis.

```http
GET /api/v1/analyses/{analysis_id}/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "status": "running",
  "progress": 65.5,
  "current_stage": "model_training",
  "execution_time": 420.8,
  "execution_mode": "local_cpu",
  "start_time": "2025-01-01T11:00:00Z",
  "estimated_completion": "2025-01-01T11:25:00Z",
  "stages_completed": [
    "data_validation",
    "feature_engineering",
    "model_selection"
  ],
  "current_stage_progress": 75.0,
  "message": "Training XGBoost model (8/10)"
}
```

**Status Values:**
- `created` - Analysis created, waiting to start
- `running` - Currently executing
- `completed` - Successfully finished
- `failed` - Execution failed
- `cancelled` - User cancelled execution

### Get Analysis Results

Retrieve complete results of a finished analysis.

```http
GET /api/v1/analyses/{analysis_id}/results
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "status": "completed",
  "task_type": "regression",
  "execution_time": 1247.3,
  "best_model_name": "XGBoost",
  "performance_metrics": {
    "r2_score": 0.847,
    "mae": 145.67,
    "mse": 33256.89,
    "rmse": 182.36
  },
  "model_comparison": [
    {
      "model_name": "XGBoost",
      "r2_score": 0.847,
      "mae": 145.67,
      "training_time": 67.4
    },
    {
      "model_name": "CatBoost", 
      "r2_score": 0.832,
      "mae": 156.23,
      "training_time": 89.1
    }
  ],
  "feature_importance": {
    "quantity": 0.342,
    "product_category": 0.278,
    "day_of_week": 0.198,
    "season": 0.182
  },
  "cross_validation_scores": [0.834, 0.851, 0.845, 0.839, 0.856],
  "artifacts": {
    "model_file": "https://storage.example.com/models/analysis_456.pkl",
    "feature_importance_plot": "https://storage.example.com/plots/feature_importance_456.png",
    "residual_plot": "https://storage.example.com/plots/residuals_456.png"
  },
  "completed_at": "2025-01-01T11:20:47Z"
}
```

### List User Analyses

Get a list of all analyses for the current user.

```http
GET /api/v1/analyses?page=1&size=10&status=completed&task_type=regression
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (int): Page number
- `size` (int): Items per page
- `status` (string): Filter by status
- `task_type` (string): Filter by task type
- `dataset_id` (int): Filter by dataset

**Response:**
```json
{
  "items": [
    {
      "id": "analysis_456",
      "dataset_id": 123,
      "task_type": "regression",
      "status": "completed",
      "created_at": "2025-01-01T11:00:00Z",
      "execution_time": 1247.3,
      "best_model_name": "XGBoost",
      "performance_score": 0.847
    }
  ],
  "total": 1,
  "page": 1,
  "pages": 1,
  "size": 20
}
```

### Cancel Analysis

Cancel a running analysis.

```http
POST /api/v1/analyses/{analysis_id}/cancel
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Analysis cancelled successfully",
  "cancelled_at": "2025-01-01T11:15:00Z",
  "partial_results_available": true
}
```

### Run Analysis (Background)

Start analysis execution in background mode.

```http
POST /api/v1/analyses/{analysis_id}/run
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Analysis started in background",
  "analysis_id": "analysis_456",
  "status": "running"
}
```

---

## 🔮 Predictions

### Make Single Prediction

Generate predictions using a trained model.

```http
POST /api/v1/predict/{analysis_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "features": {
    "quantity": 15,
    "product_category": "Electronics",
    "day_of_week": "Monday",
    "season": "Winter"
  },
  "include_probabilities": true,
  "include_explanations": true
}
```

**Response:**
```json
{
  "predictions": [2156.78],
  "probabilities": [0.92, 0.08],
  "model_name": "XGBoost",
  "confidence": 0.92,
  "explanations": {
    "feature_contributions": {
      "quantity": 456.32,
      "product_category": 234.56,
      "day_of_week": -89.12,
      "season": 123.45
    },
    "explanation_text": "The model predicts revenue of $2,156.78 with high confidence. Quantity (15 units) is the strongest positive factor, while Monday typically sees lower sales."
  },
  "prediction_time": 0.023
}
```

### Batch Predictions

Generate predictions for multiple records.

```http
POST /api/v1/predict/{analysis_id}/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "features": [
    {
      "quantity": 15,
      "product_category": "Electronics",
      "day_of_week": "Monday"
    },
    {
      "quantity": 8,
      "product_category": "Clothing",
      "day_of_week": "Friday"
    }
  ],
  "include_probabilities": false,
  "include_explanations": false
}
```

**Response:**
```json
{
  "predictions": [2156.78, 987.45],
  "model_name": "XGBoost",
  "batch_size": 2,
  "prediction_time": 0.045,
  "individual_results": [
    {
      "index": 0,
      "prediction": 2156.78,
      "confidence": 0.92
    },
    {
      "index": 1,
      "prediction": 987.45,
      "confidence": 0.78
    }
  ]
}
```

---

## 📈 Dashboard & Insights

### Get Dashboard Data

Retrieve comprehensive dashboard data for visualization.

```http
GET /api/v1/dashboard/{analysis_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overview": {
    "analysis_id": "analysis_456",
    "task_type": "regression",
    "model_performance": {
      "accuracy": 0.847,
      "primary_metric": "R² Score"
    },
    "dataset_info": {
      "rows": 10000,
      "features": 15,
      "target_column": "revenue"
    }
  },
  "visualizations": [
    {
      "type": "feature_importance",
      "title": "Feature Importance",
      "data": {
        "labels": ["quantity", "category", "day_of_week"],
        "values": [0.342, 0.278, 0.198]
      },
      "config": {
        "chart_type": "bar",
        "color_scheme": "blue"
      }
    },
    {
      "type": "performance_metrics",
      "title": "Model Performance",
      "data": {
        "metrics": ["R²", "MAE", "RMSE"],
        "values": [0.847, 145.67, 182.36],
        "benchmarks": [0.8, 150.0, 200.0]
      }
    }
  ],
  "insights": [
    {
      "title": "High Model Performance",
      "description": "Your model achieved excellent R² score of 0.847",
      "type": "performance",
      "importance": "high"
    }
  ],
  "summary": {
    "status": "completed",
    "execution_time": 1247.3,
    "models_trained": 8,
    "best_model": "XGBoost"
  }
}
```

### Get AI Insights

Retrieve AI-generated insights and recommendations.

```http
GET /api/v1/insights/{analysis_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "insights": [
    {
      "title": "Strong Predictive Model",
      "description": "Your XGBoost model achieved an R² score of 0.847, indicating excellent predictive capability. This means the model explains 84.7% of the variance in revenue.",
      "type": "performance",
      "importance": "high",
      "category": "model_quality"
    },
    {
      "title": "Key Revenue Drivers",
      "description": "Quantity is the most important factor (34.2% importance), followed by product category (27.8%). Focus on inventory management and product mix optimization.",
      "type": "feature_analysis",
      "importance": "high",
      "category": "business_insights"
    }
  ],
  "recommendations": [
    "Model is ready for production deployment",
    "Consider A/B testing with current business rules",
    "Monitor model performance with new data",
    "Investigate outliers in predictions above $5,000"
  ],
  "summary": "Analysis completed successfully with a high-performance model. The XGBoost algorithm achieved excellent results and identified key business drivers. Revenue is primarily driven by quantity and product category, suggesting opportunities for targeted inventory and marketing strategies."
}
```

### Get Model Explanations

Retrieve detailed model explanations and interpretability data.

```http
GET /api/v1/explanations/{analysis_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "model_name": "XGBoost",
  "feature_importance": {
    "global": {
      "quantity": 0.342,
      "product_category": 0.278,
      "day_of_week": 0.198,
      "season": 0.182
    },
    "method": "shap_values"
  },
  "local_examples": [
    {
      "sample_id": 0,
      "prediction": 2156.78,
      "actual": 2134.56,
      "explanation": {
        "quantity": 0.52,
        "product_category": 0.31,
        "day_of_week": -0.15,
        "season": 0.08
      }
    }
  ],
  "shap_values": {
    "available": true,
    "summary_plot": "https://storage.example.com/plots/shap_summary_456.png",
    "waterfall_plots": [
      "https://storage.example.com/plots/shap_waterfall_456_0.png"
    ]
  },
  "partial_dependence": {
    "available": true,
    "plots": {
      "quantity": "https://storage.example.com/plots/pdp_quantity_456.png",
      "product_category": "https://storage.example.com/plots/pdp_category_456.png"
    }
  }
}
```

---

## 📊 Performance Metrics

### Get Performance Metrics

Retrieve detailed performance metrics for an analysis.

```http
GET /api/v1/metrics/{analysis_id}
Authorization: Bearer <token>
```

**Response (Regression):**
```json
{
  "analysis_id": "analysis_456",
  "task_type": "regression",
  "metrics": {
    "r2_score": 0.847,
    "mae": 145.67,
    "mse": 33256.89,
    "rmse": 182.36,
    "mape": 8.34,
    "explained_variance": 0.851
  },
  "cross_validation": {
    "cv_scores": [0.834, 0.851, 0.845, 0.839, 0.856],
    "cv_mean": 0.845,
    "cv_std": 0.008,
    "folds": 5
  },
  "residual_analysis": {
    "residuals_normal": true,
    "residuals_plot": "https://storage.example.com/plots/residuals_456.png",
    "qq_plot": "https://storage.example.com/plots/qq_456.png"
  },
  "feature_statistics": {
    "num_features_used": 12,
    "num_features_engineered": 8,
    "feature_selection_method": "recursive_elimination"
  }
}
```

**Response (Classification):**
```json
{
  "analysis_id": "analysis_789",
  "task_type": "classification",
  "metrics": {
    "accuracy": 0.923,
    "precision": 0.891,
    "recall": 0.945,
    "f1_score": 0.917,
    "auc_roc": 0.967,
    "auc_pr": 0.934,
    "log_loss": 0.234
  },
  "confusion_matrix": {
    "matrix": [[850, 45], [32, 573]],
    "labels": ["Class 0", "Class 1"],
    "normalized": [[0.948, 0.052], [0.053, 0.947]]
  },
  "classification_report": {
    "Class 0": {
      "precision": 0.964,
      "recall": 0.948,
      "f1-score": 0.956,
      "support": 895
    },
    "Class 1": {
      "precision": 0.927,
      "recall": 0.947,
      "f1-score": 0.937,
      "support": 605
    }
  },
  "roc_curve": {
    "fpr": [0.0, 0.052, 1.0],
    "tpr": [0.0, 0.947, 1.0],
    "auc": 0.967
  }
}
```

---

## 📈 Monitoring & MLOps

### Data Drift Detection

Monitor data drift in deployed models.

```http
GET /api/v1/monitoring/drift/{analysis_id}?start_date=2025-01-01&end_date=2025-01-31
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "drift_detected": true,
  "drift_score": 0.23,
  "threshold": 0.1,
  "period": {
    "start_date": "2025-01-01",
    "end_date": "2025-01-31"
  },
  "feature_drift": {
    "quantity": {
      "drift_score": 0.45,
      "drift_detected": true,
      "drift_type": "distribution_shift"
    },
    "product_category": {
      "drift_score": 0.08,
      "drift_detected": false,
      "drift_type": "none"
    }
  },
  "recommendations": [
    "Retrain model with recent data",
    "Investigate quantity distribution changes",
    "Monitor prediction accuracy closely"
  ],
  "drift_plots": {
    "overall": "https://storage.example.com/plots/drift_overall_456.png",
    "feature_distributions": "https://storage.example.com/plots/drift_features_456.png"
  }
}
```

### Performance Monitoring

Monitor model performance over time.

```http
GET /api/v1/monitoring/performance/{analysis_id}?days=30
Authorization: Bearer <token>
```

**Response:**
```json
{
  "analysis_id": "analysis_456",
  "monitoring_period": "30 days",
  "performance_trend": "stable",
  "current_performance": {
    "r2_score": 0.832,
    "mae": 156.78
  },
  "baseline_performance": {
    "r2_score": 0.847,
    "mae": 145.67
  },
  "performance_degradation": {
    "r2_score_change": -0.015,
    "mae_change": 11.11,
    "significant_degradation": false
  },
  "daily_metrics": [
    {
      "date": "2025-01-01",
      "predictions_made": 1247,
      "average_confidence": 0.89,
      "r2_score": 0.834
    }
  ],
  "alerts": [
    {
      "type": "info",
      "message": "Model performance is within acceptable range",
      "created_at": "2025-01-31T23:59:59Z"
    }
  ]
}
```

---

## ⚡ Real-time Updates

### WebSocket Connection

Connect to real-time analysis progress updates.

```javascript
const ws = new WebSocket('wss://api.example.com/ws/progress/analysis_456?token=<jwt_token>');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Progress Update:', data);
};
```

**WebSocket Messages:**
```json
{
  "type": "progress_update",
  "analysis_id": "analysis_456",
  "progress": 45.6,
  "stage": "feature_engineering",
  "message": "Applying target encoding to categorical features",
  "timestamp": "2025-01-01T11:10:30Z"
}
```

```json
{
  "type": "stage_completed",
  "analysis_id": "analysis_456", 
  "completed_stage": "data_validation",
  "next_stage": "feature_engineering",
  "progress": 25.0,
  "timestamp": "2025-01-01T11:05:00Z"
}
```

```json
{
  "type": "analysis_completed",
  "analysis_id": "analysis_456",
  "status": "completed",
  "execution_time": 1247.3,
  "best_model": "XGBoost",
  "performance": 0.847,
  "timestamp": "2025-01-01T11:20:47Z"
}
```

---

## 🏥 Health & Status

### Health Check

Check API health and service status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "version": "2.0.0",
  "services": {
    "database": "healthy",
    "ml_pipeline": "healthy",
    "cache": "healthy",
    "file_storage": "healthy"
  },
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 32.1
  }
}
```

### System Status

Detailed system status for monitoring.

```http
GET /api/v1/system/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "backend_status": "operational",
  "database_status": "healthy",
  "ml_pipeline_status": "operational",
  "services_status": {
    "data_service": "healthy",
    "ml_service": "healthy",
    "insights_service": "healthy",
    "mlops_service": "degraded"
  },
  "active_analyses": 5,
  "queue_length": 2,
  "system_load": {
    "cpu_1m": 1.45,
    "cpu_5m": 1.23,
    "cpu_15m": 1.67
  },
  "uptime": "5 days, 14 hours, 23 minutes"
}
```

### Readiness Probe

Kubernetes readiness check.

```http
GET /readiness
```

### Liveness Probe

Kubernetes liveness check.

```http
GET /liveness
```

---

## 📊 Export & Integration

### Export Results

Export analysis results in various formats.

```http
GET /api/v1/export/{analysis_id}?format=json
Authorization: Bearer <token>
```

**Query Parameters:**
- `format`: Export format (`json`, `csv`, `xlsx`, `pdf`)
- `include_plots`: Include visualization plots (`true`/`false`)
- `include_model`: Include serialized model (`true`/`false`)

**Response:**
```json
{
  "analysis_summary": {
    "id": "analysis_456",
    "status": "completed",
    "model_details": {
      "name": "XGBoost",
      "performance": 0.847
    }
  },
  "export_format": "json",
  "export_timestamp": "2025-01-01T12:00:00Z",
  "download_url": "https://storage.example.com/exports/analysis_456.json",
  "expires_at": "2025-01-08T12:00:00Z"
}
```

---

## 🚨 Error Handling

### Standard Error Response

All API errors follow a consistent format:

```json
{
  "error": true,
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid request parameters",
  "details": {
    "field": "target_column",
    "error": "Column 'revenue' not found in dataset"
  },
  "timestamp": "2025-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Request validation failed |
| 400 | `INVALID_FILE_FORMAT` | Unsupported file format |
| 401 | `AUTHENTICATION_REQUIRED` | Missing or invalid token |
| 401 | `TOKEN_EXPIRED` | JWT token has expired |
| 403 | `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| 404 | `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| 409 | `RESOURCE_CONFLICT` | Resource state conflict |
| 413 | `FILE_TOO_LARGE` | File exceeds size limit |
| 422 | `UNPROCESSABLE_ENTITY` | Request body validation failed |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

---

## 📝 Rate Limiting

### Rate Limits

| Endpoint Category | Requests per Minute | Requests per Hour |
|------------------|-------------------|-------------------|
| Authentication | 20 | 100 |
| Dataset Upload | 10 | 50 |
| Analysis Creation | 5 | 25 |
| Predictions | 100 | 1000 |
| General Endpoints | 60 | 500 |

### Rate Limit Headers

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

---

## 🔧 SDK & Code Examples

### Python SDK Example

```python
import requests
import pandas as pd

class AutoAnalystClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def upload_dataset(self, file_path, name=None):
        url = f"{self.base_url}/api/v1/datasets/upload"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name} if name else {}
            response = requests.post(url, files=files, data=data, headers=self.headers)
        return response.json()
    
    def create_analysis(self, dataset_id, target_column, task_type='classification'):
        url = f"{self.base_url}/api/v1/analyses"
        data = {
            'dataset_id': dataset_id,
            'target_column': target_column,
            'task_type': task_type
        }
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
    
    def get_results(self, analysis_id):
        url = f"{self.base_url}/api/v1/analyses/{analysis_id}/results"
        response = requests.get(url, headers=self.headers)
        return response.json()

# Usage
client = AutoAnalystClient('http://localhost:8000', 'your_jwt_token')

# Upload dataset
upload_result = client.upload_dataset('sales_data.csv', 'Monthly Sales')
dataset_id = upload_result['id']

# Create analysis
analysis_result = client.create_analysis(dataset_id, 'revenue', 'regression')
analysis_id = analysis_result['analysis_id']

# Get results (after completion)
results = client.get_results(analysis_id)
print(f"Best model: {results['best_model_name']}")
print(f"R² Score: {results['performance_metrics']['r2_score']}")
```

### JavaScript/Node.js Example

```javascript
class AutoAnalystClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async uploadDataset(formData) {
        const response = await fetch(`${this.baseUrl}/api/v1/datasets/upload`, {
            method: 'POST',
            body: formData,
            headers: { 'Authorization': this.headers.Authorization }
        });
        return await response.json();
    }

    async createAnalysis(datasetId, targetColumn, taskType = 'classification') {
        const response = await fetch(`${this.baseUrl}/api/v1/analyses`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                dataset_id: datasetId,
                target_column: targetColumn,
                task_type: taskType
            })
        });
        return await response.json();
    }

    async getResults(analysisId) {
        const response = await fetch(`${this.baseUrl}/api/v1/analyses/${analysisId}/results`, {
            headers: this.headers
        });
        return await response.json();
    }

    // WebSocket for real-time updates
    connectToProgress(analysisId, token) {
        const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/progress/${analysisId}?token=${token}`);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Progress:', data.progress, data.message);
        };
        
        return ws;
    }
}

// Usage
const client = new AutoAnalystClient('http://localhost:8000', 'your_jwt_token');

async function runAnalysis() {
    try {
        // Upload dataset
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        const uploadResult = await client.uploadDataset(formData);
        
        // Create analysis
        const analysisResult = await client.createAnalysis(
            uploadResult.id,
            'target_column',
            'regression'
        );
        
        // Connect to real-time updates
        const ws = client.connectToProgress(analysisResult.analysis_id, 'your_jwt_token');
        
        // Get results when completed
        setTimeout(async () => {
            const results = await client.getResults(analysisResult.analysis_id);
            console.log('Analysis completed:', results);
        }, 30000);
        
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### cURL Examples

```bash
# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "demo", "password": "demo123"}'

# Upload dataset
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
     -H "Authorization: Bearer <token>" \
     -F "file=@sales_data.csv" \
     -F "name=Monthly Sales Data"

# Create analysis
curl -X POST "http://localhost:8000/api/v1/analyses" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": 123,
       "target_column": "revenue",
       "task_type": "regression",
       "algorithms": ["xgboost", "catboost"]
     }'

# Get analysis results
curl -X GET "http://localhost:8000/api/v1/analyses/analysis_456/results" \
     -H "Authorization: Bearer <token>"

# Make predictions
curl -X POST "http://localhost:8000/api/v1/predict/analysis_456" \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "quantity": 15,
         "product_category": "Electronics"
       }
     }'
```

---

## 📱 Frontend Integration

### React Integration Example

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AutoAnalystDashboard = ({ token }) => {
    const [datasets, setDatasets] = useState([]);
    const [analyses, setAnalyses] = useState([]);
    const [loading, setLoading] = useState(false);

    const api = axios.create({
        baseURL: 'http://localhost:8000/api/v1',
        headers: { Authorization: `Bearer ${token}` }
    });

    useEffect(() => {
        loadDatasets();
        loadAnalyses();
    }, []);

    const loadDatasets = async () => {
        try {
            const response = await api.get('/datasets');
            setDatasets(response.data.items);
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    };

    const loadAnalyses = async () => {
        try {
            const response = await api.get('/analyses');
            setAnalyses(response.data.items);
        } catch (error) {
            console.error('Error loading analyses:', error);
        }
    };

    const uploadDataset = async (file) => {
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await api.post('/datasets/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            setDatasets([...datasets, response.data]);
        } catch (error) {
            console.error('Upload failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const startAnalysis = async (datasetId, targetColumn) => {
        try {
            const response = await api.post('/analyses', {
                dataset_id: datasetId,
                target_column: targetColumn,
                task_type: 'classification'
            });
            
            // Connect to WebSocket for real-time updates
            const ws = new WebSocket(
                `ws://localhost:8000/ws/progress/${response.data.analysis_id}?token=${token}`
            );
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Analysis progress:', data.progress);
            };
            
        } catch (error) {
            console.error('Analysis failed:', error);
        }
    };

    return (
        <div className="auto-analyst-dashboard">
            <h1>Auto Data Analyst</h1>
            
            <div className="upload-section">
                <input 
                    type="file" 
                    onChange={(e) => uploadDataset(e.target.files[0])}
                    disabled={loading}
                />
                {loading && <div>Uploading...</div>}
            </div>

            <div className="datasets-section">
                <h2>Datasets</h2>
                {datasets.map(dataset => (
                    <div key={dataset.id} className="dataset-card">
                        <h3>{dataset.name}</h3>
                        <p>Rows: {dataset.num_rows} | Columns: {dataset.num_columns}</p>
                        <button 
                            onClick={() => startAnalysis(dataset.id, 'target')}
                        >
                            Start Analysis
                        </button>
                    </div>
                ))}
            </div>

            <div className="analyses-section">
                <h2>Analyses</h2>
                {analyses.map(analysis => (
                    <div key={analysis.id} className="analysis-card">
                        <h3>Analysis {analysis.id}</h3>
                        <p>Status: {analysis.status}</p>
                        <p>Model: {analysis.best_model_name}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default AutoAnalystDashboard;
```

---

## 🔄 API Versioning

### Current Version: v1

- **Base Path**: `/api/v1`
- **Stable**: ✅ Production ready
- **Supported**: ✅ Fully supported
- **Deprecated**: ❌ Not deprecated

### Version Header

Include API version in requests:

```http
Accept: application/json; version=1
```

### Backward Compatibility

- Breaking changes will result in new API version
- Previous versions supported for 12 months minimum
- Migration guides provided for version upgrades

---

## 📚 Additional Resources

### Documentation Links

- **Interactive API Docs**: `/docs`
- **API Schema**: `/openapi.json`
- **Deployment Guide**: See `DEPLOYMENT.md`
- **Development Setup**: See `DEVELOPMENT.md`

### Support & Community

- **GitHub Issues**: Report bugs and feature requests
- **Discord Community**: Real-time support and discussions  
- **Email Support**: api-support@autoanalyst.com
- **Status Page**: status.autoanalyst.com

### Rate Limiting & Quotas

- **Free Tier**: 1000 API calls/month
- **Pro Tier**: 10,000 API calls/month
- **Enterprise**: Unlimited API calls
- **File Size**: 20GB maximum per upload

---

**🚀 Start building amazing data analysis applications with the Auto Data Analyst API!**

*For more examples and detailed guides, visit our [Developer Portal](https://developers.autoanalyst.com)*
