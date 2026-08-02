"""VisionForge Application Configuration Bridge."""

from visionforge.core.config import VisionForgeSettings, get_settings

settings: VisionForgeSettings = get_settings()

__all__ = ["VisionForgeSettings", "get_settings", "settings"]
