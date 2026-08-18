"""VisionForge Backend Configuration System."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentOption(StrEnum):
    """Supported execution environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class VisionForgeSettings(BaseSettings):
    """Application settings with environment variable and .env support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core Application Settings
    project_name: str = Field(default="VisionForge", description="Project display name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: EnvironmentOption = Field(
        default=EnvironmentOption.DEVELOPMENT,
        description="Execution environment (development, staging, production, testing)",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Networking & Server
    host: str = Field(default="0.0.0.0", description="Bind host address")
    port: int = Field(default=8000, description="Bind port number", ge=1, le=65535)
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    # Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # AI Core Infrastructure Settings
    data_dir: str = Field(
        default="~/.cache/visionforge",
        description="Root persistent storage directory for datasets, models, memory, and telemetry",
    )
    model_cache_dir: str = Field(
        default="~/.cache/visionforge/models",
        description="Directory path for cached model artifacts and checkpoints",
    )
    default_device: str = Field(
        default="auto",
        description="Default compute target device (auto, cpu, cuda, mps)",
    )
    max_cached_models: int = Field(
        default=3,
        description="Maximum number of loaded models permitted in GPU/RAM memory cache",
        ge=1,
    )

    # Optional External Integrations & Infrastructure (All optional with built-in fallbacks)
    database_url: str | None = Field(
        default=None,
        description="Optional PostgreSQL/SQL database connection string",
    )
    redis_url: str | None = Field(
        default=None,
        description="Optional Redis cache and job broker connection string",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Optional Qdrant vector database URL (e.g. http://qdrant:6333)",
    )
    neo4j_url: str | None = Field(
        default=None,
        description="Optional Neo4j graph database URL (e.g. bolt://neo4j:7687)",
    )
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="Optional MLflow tracking server URI (e.g. http://mlflow:5000)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="Optional OpenAI API key for Vision-Language model execution",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Optional Anthropic API key for Vision-Language model execution",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid uppercase level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_val = v.upper()
        if upper_val not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got '{v}'")
        return upper_val

    @property
    def is_development(self) -> bool:
        """Check if environment is development."""
        return self.environment in (EnvironmentOption.DEVELOPMENT, EnvironmentOption.TESTING)

    @property
    def is_production(self) -> bool:
        """Check if environment is production."""
        return self.environment == EnvironmentOption.PRODUCTION

    @property
    def docs_url(self) -> str | None:
        """Swagger UI path (enabled in dev/staging, optional in prod)."""
        return "/docs" if not self.is_production or self.debug else None

    @property
    def redoc_url(self) -> str | None:
        """ReDoc path (enabled in dev/staging, optional in prod)."""
        return "/redoc" if not self.is_production or self.debug else None


@lru_cache
def get_settings() -> VisionForgeSettings:
    """Return a cached singleton instance of VisionForgeSettings."""
    return VisionForgeSettings()
