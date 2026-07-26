"""Guardrails service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GUARDRAILS_HOST: str = "0.0.0.0"
    GUARDRAILS_PORT: int = 8003
    GUARDRAILS_DEBUG: bool = False


settings = Settings()
