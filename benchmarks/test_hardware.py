"""
Hardware benchmarks for Feluda.

This module contains benchmarks for the hardware components of Feluda.
Run with: pytest benchmarks/ --benchmark-only
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feluda.hardware.fpga import (
    FPGAConfig,
    FPGAFamily,
    FPGAInterface,
    FPGAVendor,
    HDLLanguage,
)
from feluda.hardware.asic import (
    ASICConfig,
    ASICDesignFlow,
    ASICInterface,
    ASICTechnology,
)


# Define some fixtures
@pytest.fixture
def fpga_config():
    """Create an FPGA configuration."""
    return FPGAConfig(
        vendor=FPGAVendor.XILINX,
        family=FPGAFamily.XILINX_ULTRASCALE,
        hdl_language=HDLLanguage.VERILOG,
        part_number="xcvu9p-flgb2104-2-i",
        clock_frequency_mhz=100.0,
    )


@pytest.fixture
def asic_config():
    """Create an ASIC configuration."""
    return ASICConfig(
        technology=ASICTechnology.TSMC_28NM,
        design_flow=ASICDesignFlow.DIGITAL,
        clock_frequency_mhz=500.0,
        supply_voltage=0.9,
        target_area_mm2=1.0,
        target_power_mw=100.0,
        pdk_path="/path/to/pdk",
        tool_paths={
            "synopsys_dc": "/path/to/dc_shell",
            "cadence_innovus": "/path/to/innovus",
        },
    )


@pytest.fixture
def simple_function():
    """Create a simple function for hardware generation."""
    def add(a, b):
        return a + b
    
    return add


# Benchmarks for FPGA
def test_fpga_config_to_dict(benchmark, fpga_config):
    """Benchmark FPGA config to_dict."""
    benchmark(fpga_config.to_dict)


def test_fpga_config_from_dict(benchmark, fpga_config):
    """Benchmark FPGA config from_dict."""
    config_dict = fpga_config.to_dict()
    benchmark(FPGAConfig.from_dict, config_dict)


def test_fpga_interface_creation(benchmark, fpga_config):
    """Benchmark FPGA interface creation."""
    benchmark(FPGAInterface, fpga_config)


def test_fpga_generate_hdl(benchmark, fpga_config, simple_function):
    """Benchmark FPGA HDL generation."""
    interface = FPGAInterface(fpga_config)
    
    benchmark(
        interface.generate_hdl,
        function=simple_function,
        input_types=[int, int],
        output_type=int,
        module_name="adder",
    )


# Benchmarks for ASIC
def test_asic_config_to_dict(benchmark, asic_config):
    """Benchmark ASIC config to_dict."""
    benchmark(asic_config.to_dict)


def test_asic_config_from_dict(benchmark, asic_config):
    """Benchmark ASIC config from_dict."""
    config_dict = asic_config.to_dict()
    benchmark(ASICConfig.from_dict, config_dict)


def test_asic_interface_creation(benchmark, asic_config):
    """Benchmark ASIC interface creation."""
    benchmark(ASICInterface, asic_config)


def test_asic_generate_rtl(benchmark, asic_config, simple_function):
    """Benchmark ASIC RTL generation."""
    interface = ASICInterface(asic_config)
    
    benchmark(
        interface.generate_rtl,
        function=simple_function,
        input_types=[int, int],
        output_type=int,
        module_name="adder",
    )
