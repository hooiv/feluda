"""
Hardware Package

This package provides hardware integration capabilities for Feluda.
"""

from feluda.hardware.asic import (
    ASICConfig,
    ASICDesignFlow,
    ASICInterface,
    ASICTechnology,
)
from feluda.hardware.fpga import (
    FPGAConfig,
    FPGAFamily,
    FPGAInterface,
    FPGAVendor,
    HDLLanguage,
)
from feluda.hardware.neuromorphic import (
    NeuromorphicBackend,
    NeuromorphicConfig,
    NeuromorphicHardware,
    NeuromorphicInterface,
    NeuromorphicSimulator,
    NeuronModel,
    SynapseModel,
)
from feluda.hardware.quantum import (
    QuantumBackend,
    QuantumConfig,
    QuantumHardware,
    QuantumInterface,
    QuantumSimulator,
)

__all__ = [
    # FPGA
    "FPGAVendor",
    "FPGAFamily",
    "HDLLanguage",
    "FPGAConfig",
    "FPGAInterface",
    
    # ASIC
    "ASICTechnology",
    "ASICDesignFlow",
    "ASICConfig",
    "ASICInterface",
    
    # Quantum
    "QuantumBackend",
    "QuantumSimulator",
    "QuantumHardware",
    "QuantumConfig",
    "QuantumInterface",
    
    # Neuromorphic
    "NeuromorphicBackend",
    "NeuromorphicSimulator",
    "NeuromorphicHardware",
    "NeuronModel",
    "SynapseModel",
    "NeuromorphicConfig",
    "NeuromorphicInterface",
]
