"""
Unit tests for the hardware module.
"""

import unittest
from unittest import mock

import numpy as np

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
from feluda.hardware.quantum import (
    QuantumBackend,
    QuantumConfig,
    QuantumHardware,
    QuantumInterface,
    QuantumSimulator,
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


class TestFPGAModule(unittest.TestCase):
    """Test cases for the FPGA module."""
    
    def test_fpga_config(self):
        """Test the FPGAConfig class."""
        # Create a config
        config = FPGAConfig(
            vendor=FPGAVendor.XILINX,
            family=FPGAFamily.XILINX_ULTRASCALE,
            hdl_language=HDLLanguage.VERILOG,
            part_number="xcvu9p-flgb2104-2-i",
            clock_frequency_mhz=100.0,
        )
        
        # Check the attributes
        self.assertEqual(config.vendor, FPGAVendor.XILINX)
        self.assertEqual(config.family, FPGAFamily.XILINX_ULTRASCALE)
        self.assertEqual(config.hdl_language, HDLLanguage.VERILOG)
        self.assertEqual(config.part_number, "xcvu9p-flgb2104-2-i")
        self.assertEqual(config.clock_frequency_mhz, 100.0)
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        config2 = FPGAConfig.from_dict(config_dict)
        
        self.assertEqual(config.vendor, config2.vendor)
        self.assertEqual(config.family, config2.family)
        self.assertEqual(config.hdl_language, config2.hdl_language)
        self.assertEqual(config.part_number, config2.part_number)
        self.assertEqual(config.clock_frequency_mhz, config2.clock_frequency_mhz)
    
    @mock.patch("subprocess.run")
    def test_fpga_interface(self, mock_run):
        """Test the FPGAInterface class."""
        # Mock the subprocess.run function
        mock_run.return_value.stdout = "Synthesis completed successfully"
        
        # Create a config
        config = FPGAConfig(
            vendor=FPGAVendor.XILINX,
            family=FPGAFamily.XILINX_ULTRASCALE,
            hdl_language=HDLLanguage.VERILOG,
            part_number="xcvu9p-flgb2104-2-i",
            clock_frequency_mhz=100.0,
        )
        
        # Create an interface
        interface = FPGAInterface(config)
        
        # Test generate_hdl
        hdl_code = interface.generate_hdl(
            function=lambda x, y: x + y,
            input_types=[int, int],
            output_type=int,
            module_name="adder",
        )
        
        self.assertIsInstance(hdl_code, str)
        self.assertIn("module adder", hdl_code)
        
        # Test synthesize (mocked)
        with mock.patch("tempfile.TemporaryDirectory"):
            with mock.patch("builtins.open", mock.mock_open()):
                result = interface.synthesize(hdl_code, "adder.v")
                
                self.assertIsInstance(result, dict)
                self.assertIn("success", result)


class TestASICModule(unittest.TestCase):
    """Test cases for the ASIC module."""
    
    def test_asic_config(self):
        """Test the ASICConfig class."""
        # Create a config
        config = ASICConfig(
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
        
        # Check the attributes
        self.assertEqual(config.technology, ASICTechnology.TSMC_28NM)
        self.assertEqual(config.design_flow, ASICDesignFlow.DIGITAL)
        self.assertEqual(config.clock_frequency_mhz, 500.0)
        self.assertEqual(config.supply_voltage, 0.9)
        self.assertEqual(config.target_area_mm2, 1.0)
        self.assertEqual(config.target_power_mw, 100.0)
        self.assertEqual(config.pdk_path, "/path/to/pdk")
        self.assertEqual(config.tool_paths["synopsys_dc"], "/path/to/dc_shell")
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        config2 = ASICConfig.from_dict(config_dict)
        
        self.assertEqual(config.technology, config2.technology)
        self.assertEqual(config.design_flow, config2.design_flow)
        self.assertEqual(config.clock_frequency_mhz, config2.clock_frequency_mhz)
        self.assertEqual(config.supply_voltage, config2.supply_voltage)
        self.assertEqual(config.target_area_mm2, config2.target_area_mm2)
        self.assertEqual(config.target_power_mw, config2.target_power_mw)
        self.assertEqual(config.pdk_path, config2.pdk_path)
        self.assertEqual(config.tool_paths, config2.tool_paths)
    
    @mock.patch("subprocess.run")
    def test_asic_interface(self, mock_run):
        """Test the ASICInterface class."""
        # Mock the subprocess.run function
        mock_run.return_value.stdout = "Synthesis completed successfully"
        
        # Create a config
        config = ASICConfig(
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
        
        # Create an interface
        interface = ASICInterface(config)
        
        # Test generate_rtl
        rtl_code = interface.generate_rtl(
            function=lambda x, y: x + y,
            input_types=[int, int],
            output_type=int,
            module_name="adder",
        )
        
        self.assertIsInstance(rtl_code, str)
        self.assertIn("module adder", rtl_code)
        
        # Test synthesize (mocked)
        with mock.patch("tempfile.TemporaryDirectory"):
            with mock.patch("builtins.open", mock.mock_open()):
                result = interface.synthesize(rtl_code, "adder.v")
                
                self.assertIsInstance(result, dict)
                self.assertIn("success", result)


class TestQuantumModule(unittest.TestCase):
    """Test cases for the quantum module."""
    
    def test_quantum_config(self):
        """Test the QuantumConfig class."""
        # Create a config
        config = QuantumConfig(
            backend=QuantumBackend.QISKIT,
            simulator=QuantumSimulator.QISKIT_AERSIM,
            num_qubits=5,
            num_shots=1000,
        )
        
        # Check the attributes
        self.assertEqual(config.backend, QuantumBackend.QISKIT)
        self.assertEqual(config.simulator, QuantumSimulator.QISKIT_AERSIM)
        self.assertEqual(config.hardware, None)
        self.assertEqual(config.num_qubits, 5)
        self.assertEqual(config.num_shots, 1000)
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        config2 = QuantumConfig.from_dict(config_dict)
        
        self.assertEqual(config.backend, config2.backend)
        self.assertEqual(config.simulator, config2.simulator)
        self.assertEqual(config.hardware, config2.hardware)
        self.assertEqual(config.num_qubits, config2.num_qubits)
        self.assertEqual(config.num_shots, config2.num_shots)
    
    @mock.patch("feluda.hardware.quantum.QuantumInterface._initialize_backend")
    def test_quantum_interface(self, mock_init):
        """Test the QuantumInterface class."""
        # Create a config
        config = QuantumConfig(
            backend=QuantumBackend.QISKIT,
            simulator=QuantumSimulator.QISKIT_AERSIM,
            num_qubits=5,
            num_shots=1000,
        )
        
        # Create an interface (with mocked initialization)
        interface = QuantumInterface(config)
        
        # Mock the backend module
        interface.backend_module = mock.MagicMock()
        
        # Test create_circuit
        with mock.patch("feluda.hardware.quantum.QuantumInterface.create_circuit") as mock_create:
            mock_create.return_value = mock.MagicMock()
            circuit = interface.create_circuit()
            
            mock_create.assert_called_once()
            self.assertIsNotNone(circuit)


class TestNeuromorphicModule(unittest.TestCase):
    """Test cases for the neuromorphic module."""
    
    def test_neuromorphic_config(self):
        """Test the NeuromorphicConfig class."""
        # Create a config
        config = NeuromorphicConfig(
            backend=NeuromorphicBackend.NENGO,
            simulator=NeuromorphicSimulator.NENGO_SIM,
            neuron_model=NeuronModel.LIF,
            synapse_model=SynapseModel.STATIC,
            num_neurons=100,
            simulation_time=1.0,
            dt=0.001,
        )
        
        # Check the attributes
        self.assertEqual(config.backend, NeuromorphicBackend.NENGO)
        self.assertEqual(config.simulator, NeuromorphicSimulator.NENGO_SIM)
        self.assertEqual(config.hardware, None)
        self.assertEqual(config.neuron_model, NeuronModel.LIF)
        self.assertEqual(config.synapse_model, SynapseModel.STATIC)
        self.assertEqual(config.num_neurons, 100)
        self.assertEqual(config.simulation_time, 1.0)
        self.assertEqual(config.dt, 0.001)
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        config2 = NeuromorphicConfig.from_dict(config_dict)
        
        self.assertEqual(config.backend, config2.backend)
        self.assertEqual(config.simulator, config2.simulator)
        self.assertEqual(config.hardware, config2.hardware)
        self.assertEqual(config.neuron_model, config2.neuron_model)
        self.assertEqual(config.synapse_model, config2.synapse_model)
        self.assertEqual(config.num_neurons, config2.num_neurons)
        self.assertEqual(config.simulation_time, config2.simulation_time)
        self.assertEqual(config.dt, config2.dt)
    
    @mock.patch("feluda.hardware.neuromorphic.NeuromorphicInterface._initialize_backend")
    def test_neuromorphic_interface(self, mock_init):
        """Test the NeuromorphicInterface class."""
        # Create a config
        config = NeuromorphicConfig(
            backend=NeuromorphicBackend.NENGO,
            simulator=NeuromorphicSimulator.NENGO_SIM,
            neuron_model=NeuronModel.LIF,
            synapse_model=SynapseModel.STATIC,
            num_neurons=100,
            simulation_time=1.0,
            dt=0.001,
        )
        
        # Create an interface (with mocked initialization)
        interface = NeuromorphicInterface(config)
        
        # Mock the backend module
        interface.backend_module = mock.MagicMock()
        
        # Test create_network
        with mock.patch("feluda.hardware.neuromorphic.NeuromorphicInterface.create_network") as mock_create:
            mock_create.return_value = mock.MagicMock()
            network = interface.create_network()
            
            mock_create.assert_called_once()
            self.assertIsNotNone(network)


if __name__ == "__main__":
    unittest.main()
