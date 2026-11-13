"""Environment configuration; domain integrations are not active in this scaffold."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "job-service"
    app_env: str = "development"
    celery_broker_url: str = "redis://redis:6379/0"

    model_config = SettingsConfigDict(env_prefix="ADA_", extra="ignore")
