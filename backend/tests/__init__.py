"""
Test Package for Auto-Analyst Backend

This package contains comprehensive test suites for the Auto-Analyst backend application,
providing complete coverage for API endpoints, services, ML pipelines, and core functionality.

The test package is organized into focused modules that ensure the reliability, performance,
and security of the Auto-Analyst platform through automated testing.

Test Structure:
- API Tests: FastAPI endpoint testing with authentication and validation
- Service Tests: Business logic layer testing with mocking and integration
- ML Pipeline Tests: Machine learning workflow testing and validation
- Integration Tests: End-to-end workflow testing across components
- Performance Tests: Load testing and performance benchmarking
- Security Tests: Input validation, authentication, and vulnerability testing

Test Categories:
- Unit Tests: Individual component functionality testing
- Integration Tests: Component interaction and data flow testing
- End-to-End Tests: Complete user workflow testing
- Performance Tests: Response time and scalability validation
- Security Tests: Authentication, authorization, and input validation

Configuration:
Test configuration and fixtures are centralized in conftest.py, providing:
- FastAPI test client setup
- Database session management
- Authentication mocking
- Sample data generation
- Service mocking utilities
- Temporary file management

Usage:
    # Run all tests
    pytest
    
    # Run specific test module
    pytest tests/test_api.py
    
    # Run with coverage
    pytest --cov=backend --cov-report=html
    
    # Run specific test categories
    pytest -m "api"          # API tests only
    pytest -m "integration"  # Integration tests only
    pytest -m "performance"  # Performance tests only

Dependencies:
- pytest: Core testing framework
- pytest-asyncio: Async test support
- pytest-cov: Coverage reporting
- FastAPI TestClient: API endpoint testing
- SQLAlchemy: Database testing utilities

Best Practices:
- Tests are isolated and independent
- Fixtures provide reusable test data and configurations
- Mocking is used to isolate units under test
- Performance tests validate response time requirements
- Security tests ensure proper input validation and authentication

Note:
This package uses pytest for test discovery and execution. Tests are automatically
discovered based on naming conventions (test_*.py or *_test.py files).
No manual test registration is required.

For detailed test configuration and fixture documentation, see conftest.py.
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Auto-Analyst Backend Team"
__description__ = "Comprehensive test suite for Auto-Analyst backend"

# Package-level constants
TEST_PACKAGE_NAME = "backend.tests"
PYTEST_MIN_VERSION = "7.0.0"

# Test configuration constants
DEFAULT_TEST_TIMEOUT = 30  # seconds
MAX_CONCURRENT_TESTS = 4
TEST_DATA_RETENTION_DAYS = 1

# Export test utilities if needed (keeping minimal)
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "TEST_PACKAGE_NAME"
]

# No side effects - this file should not execute any test code
# pytest will handle test discovery and execution automatically
