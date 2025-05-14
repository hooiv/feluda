"""
Hardware Acceleration Module

This module provides hardware acceleration hooks for Feluda.
"""

import importlib
import logging
import os
import platform
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

log = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class HardwareType(str, Enum):
    """Enum for hardware types."""
    
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    FPGA = "fpga"
    ASIC = "asic"
    NPU = "npu"
    UNKNOWN = "unknown"


class HardwareVendor(str, Enum):
    """Enum for hardware vendors."""
    
    INTEL = "intel"
    AMD = "amd"
    NVIDIA = "nvidia"
    GOOGLE = "google"
    XILINX = "xilinx"
    ALTERA = "altera"
    ARM = "arm"
    APPLE = "apple"
    QUALCOMM = "qualcomm"
    UNKNOWN = "unknown"


class HardwareInfo:
    """
    Information about the hardware.
    
    This class provides information about the hardware, such as the type, vendor,
    model, and capabilities.
    """
    
    def __init__(
        self,
        hardware_type: HardwareType,
        vendor: HardwareVendor,
        model: str,
        capabilities: Dict[str, Any],
    ):
        """
        Initialize a HardwareInfo.
        
        Args:
            hardware_type: The type of hardware.
            vendor: The vendor of the hardware.
            model: The model of the hardware.
            capabilities: The capabilities of the hardware.
        """
        self.hardware_type = hardware_type
        self.vendor = vendor
        self.model = model
        self.capabilities = capabilities
    
    def __str__(self) -> str:
        """
        Get a string representation of the hardware info.
        
        Returns:
            A string representation of the hardware info.
        """
        return f"{self.vendor} {self.model} ({self.hardware_type})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the hardware info to a dictionary.
        
        Returns:
            A dictionary representation of the hardware info.
        """
        return {
            "hardware_type": self.hardware_type,
            "vendor": self.vendor,
            "model": self.model,
            "capabilities": self.capabilities,
        }


class HardwareProfile:
    """
    Profile of available hardware.
    
    This class provides information about the available hardware and their capabilities.
    """
    
    def __init__(self):
        """Initialize a HardwareProfile."""
        self.hardware_info: Dict[str, HardwareInfo] = {}
        self.default_hardware: Optional[str] = None
        self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect the available hardware."""
        # Detect CPU
        self._detect_cpu()
        
        # Detect GPU
        self._detect_gpu()
        
        # Detect TPU
        self._detect_tpu()
        
        # Detect other hardware
        self._detect_other_hardware()
        
        # Set default hardware
        if "cuda" in self.hardware_info:
            self.default_hardware = "cuda"
        elif "cpu" in self.hardware_info:
            self.default_hardware = "cpu"
    
    def _detect_cpu(self) -> None:
        """Detect the CPU."""
        try:
            import cpuinfo
            
            info = cpuinfo.get_cpu_info()
            vendor = info.get("vendor_id", "unknown").lower()
            model = info.get("brand_raw", "unknown")
            
            if "intel" in vendor:
                vendor = HardwareVendor.INTEL
            elif "amd" in vendor:
                vendor = HardwareVendor.AMD
            elif "arm" in vendor:
                vendor = HardwareVendor.ARM
            elif "apple" in vendor:
                vendor = HardwareVendor.APPLE
            else:
                vendor = HardwareVendor.UNKNOWN
            
            capabilities = {
                "cores": info.get("count", 1),
                "architecture": info.get("arch", "unknown"),
                "flags": info.get("flags", []),
            }
            
            self.hardware_info["cpu"] = HardwareInfo(
                hardware_type=HardwareType.CPU,
                vendor=vendor,
                model=model,
                capabilities=capabilities,
            )
            
        except ImportError:
            # Fallback to platform module
            processor = platform.processor()
            system = platform.system()
            
            if "intel" in processor.lower():
                vendor = HardwareVendor.INTEL
            elif "amd" in processor.lower():
                vendor = HardwareVendor.AMD
            else:
                vendor = HardwareVendor.UNKNOWN
            
            self.hardware_info["cpu"] = HardwareInfo(
                hardware_type=HardwareType.CPU,
                vendor=vendor,
                model=processor,
                capabilities={"system": system},
            )
    
    def _detect_gpu(self) -> None:
        """Detect the GPU."""
        # Try to detect CUDA GPUs
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    capabilities = {
                        "compute_capability": torch.cuda.get_device_capability(i),
                        "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    }
                    
                    self.hardware_info[f"cuda:{i}"] = HardwareInfo(
                        hardware_type=HardwareType.GPU,
                        vendor=HardwareVendor.NVIDIA,
                        model=device_name,
                        capabilities=capabilities,
                    )
        except ImportError:
            pass
        
        # Try to detect ROCm GPUs
        try:
            import torch
            
            if hasattr(torch, "hip") and torch.hip.is_available():
                device_count = torch.hip.device_count()
                
                for i in range(device_count):
                    device_name = torch.hip.get_device_name(i)
                    capabilities = {
                        "total_memory": torch.hip.get_device_properties(i).total_memory,
                    }
                    
                    self.hardware_info[f"rocm:{i}"] = HardwareInfo(
                        hardware_type=HardwareType.GPU,
                        vendor=HardwareVendor.AMD,
                        model=device_name,
                        capabilities=capabilities,
                    )
        except (ImportError, AttributeError):
            pass
    
    def _detect_tpu(self) -> None:
        """Detect the TPU."""
        try:
            import torch_xla.core.xla_model as xm
            
            device = xm.xla_device()
            device_type = str(device.type)
            
            if "xla" in device_type:
                self.hardware_info["tpu"] = HardwareInfo(
                    hardware_type=HardwareType.TPU,
                    vendor=HardwareVendor.GOOGLE,
                    model="TPU",
                    capabilities={"device_type": device_type},
                )
        except ImportError:
            pass
    
    def _detect_other_hardware(self) -> None:
        """Detect other hardware."""
        # This is a placeholder for detecting other hardware types
        # such as FPGAs, ASICs, and NPUs
        pass
    
    def get_hardware_info(self, device: Optional[str] = None) -> Optional[HardwareInfo]:
        """
        Get information about a specific hardware device.
        
        Args:
            device: The device to get information about. If None, the default device is used.
            
        Returns:
            Information about the device, or None if the device is not available.
        """
        if device is None:
            device = self.default_hardware
        
        if device is None:
            return None
        
        return self.hardware_info.get(device)
    
    def get_available_devices(self) -> List[str]:
        """
        Get a list of available devices.
        
        Returns:
            A list of available devices.
        """
        return list(self.hardware_info.keys())
    
    def get_default_device(self) -> Optional[str]:
        """
        Get the default device.
        
        Returns:
            The default device, or None if no default device is set.
        """
        return self.default_hardware
    
    def set_default_device(self, device: str) -> None:
        """
        Set the default device.
        
        Args:
            device: The device to set as the default.
            
        Raises:
            ValueError: If the device is not available.
        """
        if device not in self.hardware_info:
            raise ValueError(f"Device {device} is not available")
        
        self.default_hardware = device


# Singleton instance of HardwareProfile
_hardware_profile: Optional[HardwareProfile] = None


def get_hardware_profile() -> HardwareProfile:
    """
    Get the hardware profile.
    
    Returns:
        The hardware profile.
    """
    global _hardware_profile
    
    if _hardware_profile is None:
        _hardware_profile = HardwareProfile()
    
    return _hardware_profile
