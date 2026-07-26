"""Image Analyzer service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    IMAGE_ANALYZER_HOST: str = "0.0.0.0"
    IMAGE_ANALYZER_PORT: int = 8002
    IMAGE_ANALYZER_DEBUG: bool = False

    MAX_IMAGE_MB: int = 15
    ALLOWED_IMAGE_EXTENSIONS: str = "png,jpg,jpeg,webp,gif"

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ext.strip().lower().lstrip(".")
            for ext in self.ALLOWED_IMAGE_EXTENSIONS.split(",")
            if ext.strip()
        }


settings = Settings()
