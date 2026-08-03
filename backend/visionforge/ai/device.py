"""VisionForge Device Management & Hardware Capabilities Abstraction."""

import os
import platform
from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Hardware compute target backend classification."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon Metal Performance Shaders
    ROCM = "rocm"  # AMD ROCm
    TPU = "tpu"


class HardwareCapabilities(BaseModel):
    """Structured report of host compute device availability and hardware specs."""

    available_devices: list[DeviceType] = Field(
        default_factory=lambda: [DeviceType.CPU],
        description="List of detected compute backends",
    )
    optimal_device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Recommended default compute device target for current hardware",
    )
    cpu_cores: int = Field(default=1, description="Total physical/logical CPU core count")
    has_cuda: bool = Field(
        default=False, description="True if NVIDIA CUDA GPU acceleration is available"
    )
    has_mps: bool = Field(
        default=False, description="True if Apple Silicon Metal MPS acceleration is available"
    )
    platform_info: str = Field(description="Operating system and CPU architecture summary")


class DeviceManager:
    """Hardware abstraction manager for inspecting available compute backends."""

    def __init__(self) -> None:
        self._capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> HardwareCapabilities:
        available: list[DeviceType] = [DeviceType.CPU]
        cpu_cores = os.cpu_count() or 1
        platform_info = platform.platform()

        has_cuda = False
        has_mps = False

        # 1. Inspect Apple Silicon MPS (macOS arm64)
        if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
            has_mps = True
            available.append(DeviceType.MPS)

        # 2. Inspect CUDA availability (safely check torch if imported or sys)
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                has_cuda = True
                available.append(DeviceType.CUDA)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                has_mps = True
                if DeviceType.MPS not in available:
                    available.append(DeviceType.MPS)
        except ImportError:
            # PyTorch not installed in environment; rely on hardware architecture inspection
            pass

        # Determine optimal default compute target
        if has_cuda:
            optimal = DeviceType.CUDA
        elif has_mps:
            optimal = DeviceType.MPS
        else:
            optimal = DeviceType.CPU

        return HardwareCapabilities(
            available_devices=available,
            optimal_device=optimal,
            cpu_cores=cpu_cores,
            has_cuda=has_cuda,
            has_mps=has_mps,
            platform_info=platform_info,
        )

    def get_hardware_capabilities(self) -> HardwareCapabilities:
        """Return the detected hardware capabilities specification."""
        return self._capabilities

    def get_available_devices(self) -> list[DeviceType]:
        """Return list of supported hardware compute devices on this host."""
        return self._capabilities.available_devices

    def get_optimal_device(self) -> DeviceType:
        """Return the optimal compute device target for this host."""
        return self._capabilities.optimal_device

    def resolve_device(self, requested_device: str = "auto") -> DeviceType:
        """Resolve 'auto' or a requested device string to a valid available DeviceType.

        If requested device is unavailable, falls back to optimal available device.
        """
        clean_req = requested_device.strip().lower()

        if clean_req in ("auto", "default"):
            return self.get_optimal_device()

        try:
            target = DeviceType(clean_req)
            if target in self.get_available_devices():
                return target
        except ValueError:
            pass

        return self.get_optimal_device()


@lru_cache
def get_device_manager() -> DeviceManager:
    """Return a cached singleton instance of DeviceManager."""
    return DeviceManager()
