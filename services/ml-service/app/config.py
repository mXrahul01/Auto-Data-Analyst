"""Environment configuration; domain integrations are not active in this scaffold."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "ml-service"
    app_env: str = "development"

    model_config = SettingsConfigDict(env_prefix="ADA_", extra="ignore")
