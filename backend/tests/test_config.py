"""Test application configuration system."""

from visionforge.core.config import EnvironmentOption, VisionForgeSettings, get_settings


def test_default_settings():
    """Verify settings defaults are populated correctly."""
    settings = VisionForgeSettings()
    assert settings.project_name == "VisionForge"
    assert settings.version == "0.1.0"
    assert settings.environment == EnvironmentOption.DEVELOPMENT
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.is_development is True
    assert settings.is_production is False
    assert settings.docs_url == "/docs"


def test_singleton_settings_getter():
    """Verify get_settings returns cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_production_environment_properties():
    """Verify property behaviors in production mode."""
    prod_settings = VisionForgeSettings(environment=EnvironmentOption.PRODUCTION, debug=False)
    assert prod_settings.is_production is True
    assert prod_settings.is_development is False
    assert prod_settings.docs_url is None
