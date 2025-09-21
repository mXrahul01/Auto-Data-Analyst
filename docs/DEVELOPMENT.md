# Development Guide 👨‍💻

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PyTest](https://img.shields.io/badge/PyTest-7.0+-red.svg)](https://pytest.org)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-Enabled-brightgreen.svg)](https://pre-commit.com)

## 🌟 Overview

This comprehensive development guide provides everything needed to contribute to the Auto Data Analyst project, from initial setup to advanced development patterns. Whether you're a new contributor or an experienced developer, this guide will help you get productive quickly while following best practices.

### 🎯 Development Principles

- **🧪 Test-Driven Development** - Write tests first, code second
- **📝 Documentation-First** - Document before implementing
- **🔄 Continuous Integration** - Automated testing and deployment
- **🎨 Clean Code** - Readable, maintainable, and efficient code
- **🔒 Security-First** - Security considerations in every decision
- **⚡ Performance-Aware** - Optimize for scalability and efficiency

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites Checklist

- [ ] **Python 3.9+** installed
- [ ] **Git** configured with your GitHub account
- [ ] **VS Code** or preferred IDE
- [ ] **Docker** (optional, for containerized development)
- [ ] **PostgreSQL** (optional, SQLite works for development)

### One-Command Setup

```bash
# Clone and setup in one go
curl -sSL https://raw.githubusercontent.com/mXrahul01/Auto-Data-Analyst/main/scripts/dev-setup.sh | bash
```

### Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/mXrahul01/Auto-Data-Analyst.git
cd Auto-Data-Analyst

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your local configuration

# 5. Initialize database
alembic upgrade head

# 6. Install pre-commit hooks
pre-commit install

# 7. Run tests to verify setup
pytest

# 8. Start development server
uvicorn main:app --reload --log-level debug
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check interactive docs
open http://localhost:8000/docs

# Run test suite
pytest --cov=backend

# Check code quality
flake8 backend/
black --check backend/
mypy backend/
```

---

## 🏗️ Project Architecture

### Directory Structure

```
Auto-Data-Analyst/
├── 📁 backend/                 # Main backend application
│   ├── 📁 models/             # Database models & schemas
│   │   ├── database.py        # SQLAlchemy ORM models
│   │   └── schemas.py         # Pydantic validation schemas
│   ├── 📁 services/           # Business logic layer
│   │   ├── data_service.py    # Data processing service
│   │   ├── ml_service.py      # ML pipeline orchestration
│   │   ├── insights_service.py # Business insights service
│   │   └── mlops_service.py   # MLOps and monitoring
│   ├── 📁 ml/                 # Machine learning modules
│   │   ├── auto_pipeline.py   # Automated ML pipeline
│   │   ├── model_selection.py # Model selection & tuning
│   │   ├── feature_engineering.py # Feature processing
│   │   ├── tabular_models.py  # Tabular data models
│   │   ├── timeseries_models.py # Time series models
│   │   ├── text_models.py     # NLP and text analysis
│   │   ├── deep_models.py     # Deep learning models
│   │   ├── ensemble_models.py # Ensemble methods
│   │   ├── clustering_models.py # Clustering algorithms
│   │   ├── anomaly_models.py  # Anomaly detection
│   │   ├── recommender_models.py # Recommendation systems
│   │   ├── evaluation.py      # Model evaluation
│   │   ├── explainer.py       # Model interpretability
│   │   └── drift_detection.py # Data drift monitoring
│   ├── 📁 tasks/              # Background task processing
│   │   ├── training_tasks.py  # Model training tasks
│   │   ├── data_processing_tasks.py # Data processing
│   │   └── cleanup_tasks.py   # System maintenance
│   ├── 📁 utils/              # Utility functions
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── validation.py      # Input validation
│   │   └── monitoring.py      # System monitoring
│   ├── 📁 tests/              # Test suite
│   │   ├── test_api.py        # API endpoint tests
│   │   ├── test_ml.py         # ML pipeline tests
│   │   └── conftest.py        # Test configuration
│   ├── config.py              # Configuration management
│   └── main.py                # FastAPI application entry
├── 📁 alembic/                # Database migrations
│   └── 📁 versions/           # Migration files
├── 📁 docs/                   # Documentation
│   ├── API.md                 # API documentation
│   ├── DEPLOYMENT.md          # Deployment guide
│   └── DEVELOPMENT.md         # This file
├── 📁 scripts/                # Development scripts
│   ├── dev-setup.sh           # Development setup
│   ├── run-tests.sh           # Test runner
│   └── deploy.sh              # Deployment script
├── 📁 docker/                 # Docker configurations
│   ├── Dockerfile             # Production container
│   ├── Dockerfile.dev         # Development container
│   └── docker-compose.yml     # Multi-service setup
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml             # Python project configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
├── .github/                   # GitHub workflows
│   └── workflows/             # CI/CD pipelines
├── README.md                  # Project overview
├── .env.example               # Environment template
└── .gitignore                 # Git ignore rules
```

### Architecture Patterns

#### Service Layer Pattern
```python
# services/base_service.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseService(ABC):
    """Base service class with common functionality"""
    
    def __init__(self, db_session, cache_manager=None):
        self.db = db_session
        self.cache = cache_manager
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Main processing method - implement in subclasses"""
        pass

# services/data_service.py
class DataService(BaseService):
    """Data processing and management service"""
    
    async def process(self, file_path: str, config: Dict) -> Dict:
        # Implementation here
        pass
    
    async def load_dataset(self, file_path: str) -> pd.DataFrame:
        # Load and validate dataset
        pass
    
    async def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Data preprocessing pipeline
        pass
```

#### Repository Pattern
```python
# models/repository.py
from typing import Generic, TypeVar, Type, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import DeclarativeMeta

ModelType = TypeVar("ModelType", bound=DeclarativeMeta)

class BaseRepository(Generic[ModelType]):
    """Generic repository for database operations"""
    
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, obj_in: dict) -> ModelType:
        obj = self.model(**obj_in)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj
    
    def update(self, id: int, obj_in: dict) -> Optional[ModelType]:
        obj = self.get(id)
        if obj:
            for field, value in obj_in.items():
                setattr(obj, field, value)
            self.db.commit()
            self.db.refresh(obj)
        return obj
    
    def delete(self, id: int) -> bool:
        obj = self.get(id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
            return True
        return False
```

---

## 🧪 Testing Strategy

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── test_services/
│   ├── test_models/
│   └── test_utils/
├── integration/       # Integration tests (slower, with dependencies)
│   ├── test_api/
│   ├── test_database/
│   └── test_ml_pipeline/
├── e2e/              # End-to-end tests (full workflow)
│   └── test_complete_analysis/
├── performance/      # Performance and load tests
│   └── test_benchmarks/
├── fixtures/         # Test data and fixtures
│   ├── sample_datasets/
│   └── mock_responses/
└── conftest.py       # Global test configuration
```

### Writing Tests

#### Unit Test Example
```python
# tests/unit/test_services/test_data_service.py
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
from backend.services.data_service import DataService

class TestDataService:
    @pytest.fixture
    def mock_db_session(self):
        return Mock()
    
    @pytest.fixture  
    def data_service(self, mock_db_session):
        return DataService(db_session=mock_db_session)
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'], 
            'target': [0, 1, 0, 1, 0]
        })
    
    @pytest.mark.asyncio
    async def test_load_dataset_success(self, data_service, tmp_path):
        # Arrange
        test_file = tmp_path / "test.csv"
        test_data = "feature1,feature2,target\n1,A,0\n2,B,1\n"
        test_file.write_text(test_data)
        
        # Act
        result = await data_service.load_dataset(str(test_file))
        
        # Assert
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 2
        assert list(result.data.columns) == ['feature1', 'feature2', 'target']
        assert result.info.num_rows == 2
        assert result.info.num_columns == 3
    
    @pytest.mark.asyncio
    async def test_load_dataset_file_not_found(self, data_service):
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            await data_service.load_dataset("nonexistent.csv")
    
    @pytest.mark.asyncio
    async def test_preprocess_data(self, data_service, sample_dataframe):
        # Arrange
        config = {
            'handle_missing': True,
            'encode_categorical': True,
            'scale_numeric': True
        }
        
        # Act
        result = await data_service.preprocess_data(sample_dataframe, config)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        # Add more specific assertions based on preprocessing logic
```

#### Integration Test Example
```python
# tests/integration/test_api/test_dataset_endpoints.py
import pytest
import tempfile
import io
from fastapi.testclient import TestClient
from backend.main import app

class TestDatasetEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        # Mock authentication for tests
        return {"Authorization": "Bearer test-token"}
    
    @pytest.fixture
    def sample_csv_file(self):
        csv_content = "feature1,feature2,target\n1,A,0\n2,B,1\n3,A,0\n"
        return ("test.csv", io.StringIO(csv_content), "text/csv")
    
    def test_upload_dataset_success(self, client, auth_headers, sample_csv_file):
        # Arrange
        filename, content, content_type = sample_csv_file
        files = {"file": (filename, content.getvalue(), content_type)}
        
        # Act
        response = client.post(
            "/api/v1/datasets/upload",
            files=files,
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 201
        result = response.json()
        assert "id" in result
        assert result["name"] == filename
        assert result["status"] in ["processing", "processed"]
    
    def test_get_datasets_list(self, client, auth_headers):
        # Act
        response = client.get("/api/v1/datasets", headers=auth_headers)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert "items" in result
        assert "total" in result
        assert isinstance(result["items"], list)
    
    def test_get_dataset_details(self, client, auth_headers):
        # Arrange - First upload a dataset
        csv_content = "a,b,c\n1,2,3\n4,5,6\n"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        upload_response = client.post(
            "/api/v1/datasets/upload",
            files=files,
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Act
        response = client.get(f"/api/v1/datasets/{dataset_id}", headers=auth_headers)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == dataset_id
        assert "column_names" in result
        assert "column_types" in result
```

#### Performance Test Example  
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import asyncio
import pandas as pd
from backend.services.ml_service import MLService

class TestPerformanceBenchmarks:
    @pytest.fixture
    def large_dataset(self):
        # Create a large dataset for performance testing
        return pd.DataFrame({
            'feature1': range(100000),
            'feature2': ['A', 'B', 'C'] * 33334,
            'target': [0, 1] * 50000
        })
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_analysis_performance(self, large_dataset):
        # Arrange
        ml_service = MLService()
        config = {
            'task_type': 'classification',
            'max_time': 300,  # 5 minutes max
            'algorithms': ['xgboost', 'catboost']
        }
        
        # Act
        start_time = time.time()
        result = await ml_service.create_analysis(large_dataset, config)
        execution_time = time.time() - start_time
        
        # Assert performance requirements
        assert execution_time < 300  # Should complete within 5 minutes
        assert result.best_model_name is not None
        assert result.performance_metrics['accuracy'] > 0.7  # Minimum accuracy
    
    @pytest.mark.performance
    def test_concurrent_requests(self, client, auth_headers):
        """Test API can handle concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            response = client.get("/api/v1/datasets", headers=auth_headers)
            return response.status_code
        
        # Act - Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # Assert - All requests should succeed
        assert all(status == 200 for status in results)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only  
pytest -m performance       # Performance tests only
pytest -m "not slow"        # Skip slow tests

# Run with coverage
pytest --cov=backend --cov-report=html --cov-report=term

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_services/test_data_service.py

# Run specific test method
pytest tests/unit/test_services/test_data_service.py::TestDataService::test_load_dataset_success

# Run tests matching pattern
pytest -k "test_dataset"

# Run tests with debugging
pytest --pdb              # Drop into debugger on failure
pytest --pdb-trace        # Drop into debugger immediately
```

---

## 🎨 Code Quality & Standards

### Code Style Configuration

#### pyproject.toml
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["backend"]
known_third_party = ["fastapi", "sqlalchemy", "pandas", "numpy"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "slow: Slow running tests",
    "api: API endpoint tests",
    "ml: Machine learning tests",
    "security: Security tests"
]

[tool.coverage.run]
source = ["backend"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*"
]

[tool.coverage.report] 
precision = 2
show_missing = true
skip_covered = false

[tool.flake8]
max-line-length = 88
extend-ignore = [
    "E203",  # whitespace before ':'
    "E501",  # line too long (handled by black)
    "W503",  # line break before binary operator
]
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv"
]
```

#### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings, flake8-typing-imports]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-redis]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint-docker
        args: ["--ignore", "DL3008", "--ignore", "DL3009"]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ["--tb=short", "-x", "-q"]
```

### Code Review Guidelines

#### Pull Request Template
```markdown
## 📋 Description
Brief description of changes made.

## 🔄 Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code style update (formatting, renaming)
- [ ] ♻️ Code refactoring (no functional changes)
- [ ] ⚡ Performance improvements
- [ ] 🧪 Adding tests
- [ ] 🔧 CI/CD changes

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📚 Documentation
- [ ] Code is self-documenting with clear variable names
- [ ] Complex business logic has explanatory comments
- [ ] Public APIs have docstrings
- [ ] README updated (if applicable)

## ✅ Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## 🔗 Related Issues
Fixes #(issue)

## 📸 Screenshots (if applicable)
Add screenshots to help explain your changes.

## 🔍 Additional Notes
Any additional information that reviewers should know.
```

#### Code Review Checklist
```markdown
## Code Review Checklist

### 🔍 Code Quality
- [ ] Code is readable and self-documenting
- [ ] Functions are small and focused (single responsibility)
- [ ] Variable and function names are descriptive
- [ ] No code duplication (DRY principle)
- [ ] Complex logic is properly commented
- [ ] Error handling is appropriate and comprehensive

### 🏗️ Architecture & Design
- [ ] Changes follow established patterns
- [ ] New components fit well with existing architecture
- [ ] Dependencies are minimal and justified
- [ ] Interface design is clean and intuitive
- [ ] Performance implications are considered

### 🧪 Testing
- [ ] Adequate test coverage for new functionality
- [ ] Tests are meaningful and not just for coverage
- [ ] Edge cases are covered
- [ ] Integration points are tested
- [ ] Performance tests included where relevant

### 🔒 Security
- [ ] Input validation is present
- [ ] No sensitive data in logs or responses
- [ ] Authentication/authorization properly implemented
- [ ] SQL injection protection in place
- [ ] XSS prevention measures implemented

### 📚 Documentation
- [ ] Public APIs have docstrings
- [ ] Complex algorithms are explained
- [ ] Configuration changes are documented
- [ ] Breaking changes are highlighted

### 🚀 Performance
- [ ] Database queries are optimized
- [ ] Memory usage is reasonable
- [ ] No unnecessary processing in hot paths
- [ ] Caching strategies are appropriate
- [ ] Async/await used where beneficial
```

---

## 🛠️ Development Tools & Setup

### VS Code Configuration

#### .vscode/settings.json
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.pylintEnabled": false,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    "*.egg-info": true
  }
}
```

#### .vscode/launch.json (Debugging)
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Server",
      "type": "python",
      "request": "launch",
      "program": "-m",
      "args": ["uvicorn", "main:app", "--reload", "--log-level", "debug"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "ENVIRONMENT": "development"
      }
    },
    {
      "name": "Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Debug Current Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v", "-s"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

#### .vscode/extensions.json
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.isort",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "streetsidesoftware.code-spell-checker",
    "eamodio.gitlens",
    "ms-vscode.test-adapter-converter",
    "littlefoxteam.vscode-python-test-adapter"
  ]
}
```

### Development Scripts

#### scripts/dev-setup.sh
```bash
#!/bin/bash
# Development environment setup script

set -e  # Exit on any error

echo "🚀 Setting up Auto Data Analyst development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9.0"

if ! python3 -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$required_version'.split('.'))) else 1)"; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Setting up environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your local configuration"
fi

# Initialize database
echo "🗄️ Initializing database..."
alembic upgrade head

# Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
pre-commit install

# Run initial tests
echo "🧪 Running initial tests..."
pytest --tb=short

echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: source venv/bin/activate"
echo "3. Run: uvicorn main:app --reload"
echo "4. Open: http://localhost:8000/docs"
```

#### scripts/run-tests.sh
```bash
#!/bin/bash
# Comprehensive test runner script

set -e

echo "🧪 Running Auto Data Analyst test suite..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Parse command line arguments
COVERAGE=false
VERBOSE=false
FAST=false
PERFORMANCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --fast|-f)
            FAST=true
            shift
            ;;
        --performance|-p)
            PERFORMANCE=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=backend --cov-report=html --cov-report=term"
fi

if [ "$FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow' -x"
fi

if [ "$PERFORMANCE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m performance"
fi

# Run code quality checks first
echo "🔍 Running code quality checks..."

echo "  📏 Checking code formatting with black..."
black --check backend/

echo "  🎯 Checking import sorting with isort..."
isort --check-only backend/

echo "  🔬 Running flake8 linting..."
flake8 backend/

echo "  🔍 Running mypy type checking..."
mypy backend/

echo "  🔒 Running security check with bandit..."
bandit -r backend/ -f json -o bandit-report.json || true

# Run tests
echo ""
echo "🧪 Running tests..."
eval $PYTEST_CMD

# Generate coverage report if requested
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "📊 Coverage report generated in htmlcov/index.html"
fi

echo ""
echo "✅ All tests completed successfully!"
```

#### scripts/lint.sh
```bash
#!/bin/bash
# Code quality and linting script

set -e

echo "🔍 Running code quality checks..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Format code
echo "📏 Formatting code with black..."
black backend/

echo "🎯 Sorting imports with isort..."
isort backend/

# Run linting
echo "🔬 Running flake8..."
flake8 backend/

echo "🔍 Running mypy..."
mypy backend/

echo "🔒 Running security check..."
bandit -r backend/

echo "✅ Code quality checks completed!"
```

### Database Migrations

#### Creating Migrations
```bash
# Generate a new migration
alembic revision --autogenerate -m "Add new table"

# Review the generated migration file
# Edit alembic/versions/xxx_add_new_table.py if needed

# Apply the migration
alembic upgrade head

# Downgrade if needed
alembic downgrade -1
```

#### Migration Best Practices
```python
# alembic/versions/xxx_example_migration.py
"""Add user preferences table

Revision ID: abc123
Revises: def456
Create Date: 2025-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'abc123'
down_revision = 'def456'
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Add user preferences table with proper constraints"""
    
    # Create table
    op.create_table('user_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('preferences', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('ix_user_preferences_user_id', 'user_preferences', ['user_id'])
    
    # Add check constraints if needed
    op.create_check_constraint(
        'ck_user_preferences_valid_json',
        'user_preferences', 
        "preferences IS NULL OR jsonb_typeof(preferences) = 'object'"
    )

def downgrade() -> None:
    """Remove user preferences table"""
    op.drop_table('user_preferences')
```

---

## 🚀 Performance Optimization

### Profiling Tools Setup

#### Memory Profiling
```python
# utils/profiling.py
import functools
import cProfile
import pstats
import io
from memory_profiler import profile
import psutil
import os

def profile_memory(func):
    """Decorator to profile memory usage"""
    @functools.wraps(func)
    @profile
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def profile_cpu(func):
    """Decorator to profile CPU usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Print stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            print(f"\n🔍 CPU Profile for {func.__name__}:")
            print(s.getvalue())
            
        return result
    return wrapper

def monitor_system_resources():
    """Monitor current system resource usage"""
    process = psutil.Process(os.getpid())
    
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'num_threads': process.num_threads(),
        'open_files': len(process.open_files()),
    }

# Usage example
@profile_memory
@profile_cpu
async def expensive_ml_operation(data):
    """Example of profiling an expensive operation"""
    # Your ML code here
    pass
```

#### Performance Benchmarks
```python
# tests/performance/benchmarks.py
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

class PerformanceBenchmarks:
    """Performance benchmark suite"""
    
    @pytest.fixture
    def benchmark_datasets(self):
        """Generate datasets of different sizes for benchmarking"""
        return {
            'small': pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.choice([0, 1], 1000)
            }),
            'medium': pd.DataFrame({
                'feature1': np.random.randn(50000),
                'feature2': np.random.choice(['A', 'B', 'C'], 50000),
                'target': np.random.choice([0, 1], 50000)
            }),
            'large': pd.DataFrame({
                'feature1': np.random.randn(500000),
                'feature2': np.random.choice(['A', 'B', 'C'], 500000),
                'target': np.random.choice([0, 1], 500000)
            })
        }
    
    @pytest.mark.performance
    def test_data_loading_performance(self, benchmark_datasets):
        """Benchmark data loading performance"""
        for size, df in benchmark_datasets.items():
            start_time = time.time()
            
            # Test CSV loading
            csv_path = f"/tmp/test_{size}.csv"
            df.to_csv(csv_path, index=False)
            loaded_df = pd.read_csv(csv_path)
            
            load_time = time.time() - start_time
            
            # Performance assertions
            if size == 'small':
                assert load_time < 1.0  # Small dataset should load in < 1s
            elif size == 'medium':
                assert load_time < 5.0  # Medium dataset should load in < 5s
            elif size == 'large':
                assert load_time < 20.0  # Large dataset should load in < 20s
            
            assert len(loaded_df) == len(df)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_api_performance(self, client, auth_headers):
        """Test API performance under concurrent load"""
        
        async def make_request():
            response = client.get("/api/v1/datasets", headers=auth_headers)
            return response.status_code, response.elapsed.total_seconds()
        
        # Test with increasing concurrent requests
        for concurrent_users in [1, 5, 10, 20]:
            tasks = [make_request() for _ in range(concurrent_users)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Analyze results
            status_codes = [r[0] for r in results]
            response_times = [r[1] for r in results]
            
            # Performance assertions
            assert all(code == 200 for code in status_codes)
            assert max(response_times) < 5.0  # Max response time < 5s
            assert total_time < 10.0  # Total time for all requests < 10s
            
            avg_response_time = sum(response_times) / len(response_times)
            print(f"Concurrent users: {concurrent_users}, Avg response time: {avg_response_time:.3f}s")
```

### Database Optimization

#### Query Optimization
```python
# utils/db_optimization.py
from sqlalchemy import text, func
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Database performance optimization utilities"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def analyze_slow_queries(self, min_duration_ms: int = 1000):
        """Analyze slow queries from pg_stat_statements"""
        query = text("""
            SELECT query, calls, total_time, mean_time, rows
            FROM pg_stat_statements 
            WHERE mean_time > :min_duration
            ORDER BY total_time DESC 
            LIMIT 20
        """)
        
        result = self.db.execute(query, {'min_duration': min_duration_ms})
        slow_queries = result.fetchall()
        
        for query_info in slow_queries:
            logger.warning(f"Slow query detected: {query_info.mean_time:.2f}ms avg")
            logger.warning(f"Query: {query_info.query[:100]}...")
        
        return slow_queries
    
    def optimize_table_statistics(self, table_name: str):
        """Update table statistics for better query planning"""
        self.db.execute(text(f"ANALYZE {table_name}"))
        self.db.commit()
        logger.info(f"Updated statistics for table: {table_name}")
    
    def check_index_usage(self):
        """Check index usage statistics"""
        query = text("""
            SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE idx_tup_read = 0 OR idx_tup_fetch = 0
            ORDER BY schemaname, tablename
        """)
        
        unused_indexes = self.db.execute(query).fetchall()
        
        for index_info in unused_indexes:
            logger.warning(f"Potentially unused index: {index_info.indexname}")
        
        return unused_indexes
    
    def suggest_indexes(self, table_name: str):
        """Suggest indexes based on query patterns"""
        # This is a simplified example - in practice you'd analyze query logs
        query = text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table_name
            AND column_name IN ('created_at', 'updated_at', 'user_id', 'status')
        """)
        
        columns = self.db.execute(query, {'table_name': table_name}).fetchall()
        
        suggestions = []
        for col in columns:
            if 'id' in col.column_name.lower():
                suggestions.append(f"CREATE INDEX idx_{table_name}_{col.column_name} ON {table_name} ({col.column_name});")
            elif col.column_name in ['created_at', 'updated_at']:
                suggestions.append(f"CREATE INDEX idx_{table_name}_{col.column_name} ON {table_name} ({col.column_name} DESC);")
        
        return suggestions
```

---

## 🔒 Security Development

### Security Testing

#### Security Test Cases
```python
# tests/security/test_security.py
import pytest
import jwt
from datetime import datetime, timedelta
from backend.core.security import create_access_token, verify_token

class TestSecurity:
    """Security-focused test cases"""
    
    def test_jwt_token_creation_and_validation(self):
        """Test JWT token security"""
        # Test valid token creation
        user_data = {"user_id": 123, "email": "test@example.com"}
        token = create_access_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens should be reasonably long
        
        # Test token validation
        decoded_data = verify_token(token)
        assert decoded_data["user_id"] == 123
        assert decoded_data["email"] == "test@example.com"
    
    def test_jwt_token_expiration(self):
        """Test token expiration handling"""
        # Create expired token
        expired_token = jwt.encode({
            "user_id": 123,
            "exp": datetime.utcnow() - timedelta(hours=1)
        }, "secret", algorithm="HS256")
        
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_token(expired_token)
    
    def test_sql_injection_prevention(self, client, auth_headers):
        """Test SQL injection prevention"""
        # Try various SQL injection payloads
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; SELECT * FROM users; --",
            "1; DELETE FROM datasets; --"
        ]
        
        for payload in malicious_payloads:
            # Test in query parameters
            response = client.get(f"/api/v1/datasets?search={payload}", headers=auth_headers)
            # Should not return 500 error (which might indicate SQL injection)
            assert response.status_code in [200, 400, 404]
            
            # Test in path parameters
            response = client.get(f"/api/v1/datasets/{payload}", headers=auth_headers)
            assert response.status_code in [400, 404]  # Should handle gracefully
    
    def test_xss_prevention(self, client, auth_headers):
        """Test XSS prevention"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            # Test XSS in file upload
            files = {"file": ("test.csv", f"name,value\n{payload},123\n", "text/csv")}
            response = client.post("/api/v1/datasets/upload", files=files, headers=auth_headers)
            
            # Check that payload is not reflected in response
            if response.status_code == 200:
                assert payload not in response.text
    
    def test_authentication_bypass_attempts(self, client):
        """Test authentication bypass prevention"""
        # Try accessing protected endpoints without authentication
        protected_endpoints = [
            "/api/v1/datasets",
            "/api/v1/analyses",
            "/api/v1/users/me"
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401
    
    def test_authorization_enforcement(self, client):
        """Test that users can only access their own resources"""
        # This would require setting up test users and checking access
        pass  # Implementation depends on your authorization model
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality"""
        # Make many requests rapidly
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = client.get("/api/v1/datasets", headers=auth_headers)
            responses.append(response.status_code)
        
        # Should eventually hit rate limit
        assert 429 in responses  # HTTP 429 Too Many Requests
    
    def test_file_upload_security(self, client, auth_headers):
        """Test file upload security measures"""
        # Test malicious file types
        malicious_files = [
            ("malware.exe", b"MZ\x90\x00", "application/octet-stream"),
            ("script.js", b"alert('malicious')", "application/javascript"),
            ("shell.sh", b"#!/bin/bash\nrm -rf /", "text/plain")
        ]
        
        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            response = client.post("/api/v1/datasets/upload", files=files, headers=auth_headers)
            
            # Should reject malicious files
            assert response.status_code in [400, 415]  # Bad Request or Unsupported Media Type
```

### Input Validation

#### Validation Utilities
```python
# utils/security.py
import re
import html
import bleach
from typing import Any, Dict, List, Optional
import sqlparse
from urllib.parse import urlparse

class SecurityValidator:
    """Security-focused input validation utilities"""
    
    @staticmethod
    def sanitize_html(input_text: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        if not input_text:
            return ""
        
        # Allow only safe tags and attributes
        allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
        allowed_attributes = {}
        
        cleaned = bleach.clean(
            input_text,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        # HTML escape as additional protection
        return html.escape(cleaned)
    
    @staticmethod
    def validate_sql_injection(input_text: str) -> bool:
        """Check for potential SQL injection patterns"""
        if not input_text:
            return True
        
        # Common SQL injection patterns
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)",
            r"('|\"|;|\||&)",
        ]
        
        input_upper = input_text.upper()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_upper, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file type based on extension"""
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1]
        return extension in [ext.lower().lstrip('.') for ext in allowed_extensions]
    
    @staticmethod
    def validate_file_content(file_content: bytes, max_size: int = 20 * 1024 * 1024) -> Dict[str, Any]:
        """Validate file content for security issues"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'file_info': {}
        }
        
        # Check file size
        if len(file_content) > max_size:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"File too large: {len(file_content)} bytes")
        
        # Check for executable signatures
        executable_signatures = [
            b'MZ',      # Windows PE
            b'\x7fELF',  # Linux ELF
            b'\xca\xfe\xba\xbe',  # Java class
            b'#!/bin/bash',  # Shell script
            b'#!/bin/sh',   # Shell script
        ]
        
        for signature in executable_signatures:
            if file_content.startswith(signature):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Executable file detected")
                break
        
        # Store file info
        validation_result['file_info'] = {
            'size': len(file_content),
            'has_null_bytes': b'\x00' in file_content[:1024],  # Check first 1KB
        }
        
        return validation_result
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL to prevent SSRF attacks"""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            
            # Only allow HTTP and HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Block internal/private IP ranges
            hostname = parsed.hostname
            if hostname:
                import ipaddress
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return False
                except ValueError:
                    # Not an IP address, allow domain names
                    pass
            
            return True
            
        except Exception:
            return False
```

---

## 📚 Documentation Guidelines

### Code Documentation

#### Docstring Standards
```python
def analyze_dataset(
    data: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    config: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
    """
    Analyze a dataset using automated machine learning pipeline.
    
    This function performs comprehensive data analysis including data quality
    assessment, feature engineering, model selection, training, and evaluation.
    It supports multiple task types and provides detailed insights about the
    data and model performance.
    
    Args:
        data (pd.DataFrame): The input dataset to analyze. Must contain at 
            least the target column and one feature column.
        target_column (str): Name of the target column for prediction.
            Must exist in the dataset and contain valid target values.
        task_type (str, optional): Type of machine learning task. 
            Supported values: 'classification', 'regression', 'time_series',
            'clustering', 'anomaly_detection'. Defaults to 'classification'.
        config (Dict[str, Any], optional): Configuration dictionary for
            customizing the analysis pipeline. See AnalysisConfig for
            available options. Defaults to None (uses default config).
    
    Returns:
        AnalysisResult: Object containing analysis results including:
            - best_model: The best performing model
            - performance_metrics: Dictionary of performance metrics
            - feature_importance: Feature importance scores
            - insights: Generated business insights
            - artifacts: Paths to saved artifacts (plots, models, etc.)
    
    Raises:
        ValueError: If target_column doesn't exist in data or if task_type
            is not supported.
        DataQualityError: If data quality is too poor for reliable analysis
            (e.g., >80% missing values, no variation in target).
        InsufficientDataError: If dataset is too small for the selected
            task type (minimum 50 samples for most tasks).
    
    Example:
        >>> import pandas as pd
        >>> from backend.ml.auto_pipeline import analyze_dataset
        
        >>> # Load sample data
        >>> data = pd.read_csv('sales_data.csv')
        >>> 
        >>> # Run analysis
        >>> result = analyze_dataset(
        ...     data=data,
        ...     target_column='revenue',
        ...     task_type='regression',
        ...     config={'max_time': 300, 'cv_folds': 5}
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Best model: {result.best_model.name}")
        >>> print(f"R² Score: {result.performance_metrics['r2_score']:.3f}")
        >>> print(f"Top features: {list(result.feature_importance.keys())[:5]}")
    
    Note:
        - Large datasets (>1M rows) may take significant time to process
        - For time series tasks, ensure data includes a datetime column
        - GPU acceleration is used when available for deep learning models
        
    See Also:
        AnalysisConfig: Configuration options for the analysis pipeline
        AnalysisResult: Detailed description of result object structure
        ModelEvaluator: Low-level model evaluation utilities
    
    Version:
        Added in version 2.0.0
        
    Performance:
        Time complexity: O(n * m * k) where n=samples, m=features, k=models
        Space complexity: O(n * m) for data storage plus model memory
    """
    # Implementation here
    pass
```

#### API Documentation
```python
# main.py - FastAPI endpoint documentation
@app.post(
    "/api/v1/analyses", 
    response_model=schemas.AnalysisResponse,
    status_code=201,
    tags=["Machine Learning"],
    summary="Create ML Analysis",
    description="""
    Create a new machine learning analysis on an uploaded dataset.
    
    This endpoint initiates an automated ML pipeline that includes:
    - Data validation and quality assessment  
    - Feature engineering and selection
    - Model training with multiple algorithms
    - Performance evaluation and comparison
    - Business insights generation
    
    The analysis runs asynchronously in the background. Use the returned
    analysis_id to check progress and retrieve results.
    """,
    responses={
        201: {
            "description": "Analysis created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "analysis_123",
                        "status": "created", 
                        "message": "Analysis started successfully",
                        "estimated_completion": "2025-01-01T12:30:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Target column 'revenue' not found in dataset"
                    }
                }
            }
        },
        401: {"description": "Authentication required"},
        404: {"description": "Dataset not found"},
        413: {"description": "Dataset too large"},
        422: {"description": "Validation error"}
    }
)
async def create_analysis(
    request: schemas.AnalysisRequest,
    current_user: schemas.User = Depends(get_current_user),
    ml_service: MLService = Depends(get_ml_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new ML analysis"""
    # Implementation here
    pass
```

### README Templates

#### Feature README Template
```markdown
# Feature Name

## Overview
Brief description of what this feature does and why it exists.

## Usage

### Basic Usage
```python
from backend.features.feature_name import FeatureClass

# Basic example
feature = FeatureClass()
result = feature.process(data)
```

### Advanced Usage
```python
# Advanced configuration
config = {
    'parameter1': 'value1',
    'parameter2': 42
}
feature = FeatureClass(config=config)
result = feature.process(data, additional_param='value')
```

## API Reference

### Classes

#### FeatureClass
Main class for this feature.

**Parameters:**
- `config` (dict, optional): Configuration parameters
- `debug` (bool, optional): Enable debug mode

**Methods:**
- `process(data)`: Process the input data
- `validate(data)`: Validate input data
- `get_config()`: Get current configuration

### Functions

#### helper_function(param1, param2)
Helper function description.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameter1` | str | "default" | Parameter description |
| `parameter2` | int | 42 | Another parameter |

## Examples

### Example 1: Basic Processing
```python
# Code example with explanation
```

### Example 2: Error Handling
```python
# Code example showing error handling
```

## Performance Notes

- Performance characteristics
- Memory usage considerations  
- Scalability limits

## Testing

```bash
# Run tests for this feature
pytest tests/test_feature_name.py

# Run with coverage
pytest --cov=backend.features.feature_name tests/test_feature_name.py
```

## Contributing

Guidelines specific to this feature:
- Code style requirements
- Testing requirements  
- Documentation requirements

## Changelog

### Version 2.1.0
- Added new functionality
- Fixed bug with edge case

### Version 2.0.0
- Initial implementation
```

---

## 🤝 Contributing Guidelines

### Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Auto-Data-Analyst.git
   cd Auto-Data-Analyst
   ```

2. **Set Up Development Environment**
   ```bash
   # Follow the setup instructions above
   ./scripts/dev-setup.sh
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

5. **Test Your Changes**
   ```bash
   ./scripts/run-tests.sh --coverage
   ./scripts/lint.sh
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Request reviews

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format
type(scope): description

# Types
feat:     New feature
fix:      Bug fix  
docs:     Documentation changes
style:    Code style changes (formatting, etc.)
refactor: Code refactoring
test:     Adding or updating tests
chore:    Maintenance tasks
perf:     Performance improvements
ci:       CI/CD changes

# Examples
feat(ml): add support for time series forecasting
fix(api): resolve issue with file upload validation
docs(readme): update installation instructions
test(services): add unit tests for data service
```

### Code Review Process

1. **Self Review**
   - Check your own code first
   - Run all tests and linting
   - Verify documentation is updated

2. **Peer Review**
   - At least one approval required
   - Address all feedback
   - Update based on suggestions

3. **Maintainer Review**
   - Final review by maintainers
   - Focus on architecture and design
   - Merge when approved

---

## 🆘 Troubleshooting

### Common Development Issues

#### Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to .env
echo "PYTHONPATH=$(pwd)" >> .env
```

#### Database Issues
```bash
# Problem: Database connection errors
# Solution: Check database status
systemctl status postgresql

# Reset database
alembic downgrade base
alembic upgrade head
```

#### Virtual Environment Issues  
```bash
# Problem: Package conflicts
# Solution: Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Test Failures
```bash
# Problem: Tests failing locally
# Solution: Check test isolation
pytest --lf  # Run only failed tests
pytest --tb=long  # Get detailed traceback
pytest -s  # Don't capture output
```

#### Performance Issues
```bash
# Problem: Slow development server
# Solution: Profile the application
pip install py-spy
py-spy top --pid $(pgrep -f "uvicorn")
```

### Getting Help

- **📖 Documentation**: Check existing docs first
- **🐛 GitHub Issues**: Search existing issues
- **💬 Discord**: Join our development channel
- **📧 Email**: dev-help@autoanalyst.com

---

## 📞 Community & Support

### Development Community

- **🐙 GitHub Discussions**: [github.com/mXrahul01/Auto-Data-Analyst/discussions](https://github.com/mXrahul01/Auto-Data-Analyst/discussions)
- **💬 Discord Server**: [discord.gg/autoanalyst](https://discord.gg/autoanalyst)
- **📧 Mailing List**: dev-subscribe@autoanalyst.com
- **🐦 Twitter**: [@AutoDataAnalyst](https://twitter.com/AutoDataAnalyst)

### Office Hours

Join our weekly office hours for development support:
- **When**: Every Friday, 3:00 PM UTC
- **Where**: Discord voice channel #office-hours
- **What**: Q&A, code reviews, architecture discussions

### Recognition

Contributors are recognized in:
- **Hall of Fame**: Top contributors on README
- **Release Notes**: Feature contributions highlighted
- **Conference Talks**: Speaking opportunities at events
- **Swag**: Contributor merchandise program

---

**🚀 Happy coding! Welcome to the Auto Data Analyst development community!**

*For the most up-to-date development information, visit our [Developer Portal](https://dev.autoanalyst.com)*
