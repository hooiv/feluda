"""
Tests for the ImageVecRepResNet operator (version 2).
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from feluda.exceptions import OperatorExecutionError, OperatorInitializationError, OperatorValidationError
from operators.image_vec_rep_resnet.image_vec_rep_resnet_v2 import (
    ImageVecRepResNet,
    ImageVecRepResNetParameters,
)


class TestImageVecRepResNetParameters(unittest.TestCase):
    """Tests for the ImageVecRepResNetParameters class."""
    
    def test_default_parameters(self):
        """Test default parameters."""
        params = ImageVecRepResNetParameters()
        self.assertEqual(params.model_name, "resnet50")
        self.assertTrue(params.use_pretrained)
        self.assertEqual(params.device, "auto")
        self.assertEqual(params.batch_size, 32)
        self.assertTrue(params.normalize_embeddings)
    
    def test_custom_parameters(self):
        """Test custom parameters."""
        params = ImageVecRepResNetParameters(
            model_name="resnet18",
            use_pretrained=False,
            device="cpu",
            batch_size=16,
            normalize_embeddings=False
        )
        self.assertEqual(params.model_name, "resnet18")
        self.assertFalse(params.use_pretrained)
        self.assertEqual(params.device, "cpu")
        self.assertEqual(params.batch_size, 16)
        self.assertFalse(params.normalize_embeddings)
    
    def test_invalid_model_name(self):
        """Test validation of model name."""
        with self.assertRaises(ValueError):
            ImageVecRepResNetParameters(model_name="invalid_model")
    
    def test_invalid_device(self):
        """Test validation of device."""
        with self.assertRaises(ValueError):
            ImageVecRepResNetParameters(device="invalid_device")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestImageVecRepResNetCuda(unittest.TestCase):
    """Tests for the ImageVecRepResNet operator with CUDA."""
    
    def test_auto_device_with_cuda(self):
        """Test that 'auto' device selects CUDA when available."""
        params = ImageVecRepResNetParameters(device="auto")
        self.assertEqual(params.device, "cuda")


class TestImageVecRepResNet(unittest.TestCase):
    """Tests for the ImageVecRepResNet operator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a temporary test image
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_image_path = os.path.join(cls.temp_dir.name, "test_image.jpg")
        
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures."""
        cls.temp_dir.cleanup()
    
    @patch("torchvision.models.resnet50")
    def test_initialization(self, mock_resnet):
        """Test operator initialization."""
        # Mock the ResNet model
        mock_model = MagicMock()
        mock_model.children.return_value = [MagicMock()]
        mock_resnet.return_value = mock_model
        
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Check that the model was initialized correctly
        mock_resnet.assert_called_once_with(pretrained=True)
        self.assertIsNotNone(operator.model)
        self.assertIsNotNone(operator.transform)
        self.assertEqual(operator.device, torch.device("cpu"))
    
    @patch("torchvision.models.resnet18")
    def test_initialization_with_custom_parameters(self, mock_resnet):
        """Test operator initialization with custom parameters."""
        # Mock the ResNet model
        mock_model = MagicMock()
        mock_model.children.return_value = [MagicMock()]
        mock_resnet.return_value = mock_model
        
        # Initialize the operator with custom parameters
        operator = ImageVecRepResNet(parameters={
            "model_name": "resnet18",
            "use_pretrained": False,
            "device": "cpu",
            "batch_size": 16,
            "normalize_embeddings": False
        })
        
        # Check that the model was initialized correctly
        mock_resnet.assert_called_once_with(pretrained=False)
        self.assertIsNotNone(operator.model)
        self.assertIsNotNone(operator.transform)
        self.assertEqual(operator.device, torch.device("cpu"))
    
    @patch("torchvision.models.resnet50")
    def test_initialization_failure(self, mock_resnet):
        """Test operator initialization failure."""
        # Make the model initialization fail
        mock_resnet.side_effect = RuntimeError("Model initialization failed")
        
        # Check that the operator raises an OperatorInitializationError
        with self.assertRaises(OperatorInitializationError):
            ImageVecRepResNet()
    
    @patch("operators.image_vec_rep_resnet.image_vec_rep_resnet_v2.ImageVecRepResNet._load_image")
    @patch("torch.nn.Sequential")
    @patch("torchvision.models.resnet50")
    def test_execute_with_string_path(self, mock_resnet, mock_sequential, mock_load_image):
        """Test operator execution with a string path."""
        # Mock the model and its output
        mock_model = MagicMock()
        mock_model.children.return_value = [MagicMock()]
        mock_resnet.return_value = mock_model
        
        # Mock the model output
        mock_output = MagicMock()
        mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_sequential.return_value.return_value = mock_output
        
        # Mock the image loading
        mock_image = MagicMock()
        mock_image.unsqueeze.return_value = MagicMock()
        mock_load_image.return_value = mock_image
        
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Execute the operator
        result = operator.run(self.test_image_path)
        
        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        
        # Check that the image was loaded
        mock_load_image.assert_called_once_with(self.test_image_path)
    
    @patch("operators.image_vec_rep_resnet.image_vec_rep_resnet_v2.ImageVecRepResNet._load_image")
    @patch("torch.nn.Sequential")
    @patch("torchvision.models.resnet50")
    def test_execute_with_dict(self, mock_resnet, mock_sequential, mock_load_image):
        """Test operator execution with a dictionary."""
        # Mock the model and its output
        mock_model = MagicMock()
        mock_model.children.return_value = [MagicMock()]
        mock_resnet.return_value = mock_model
        
        # Mock the model output
        mock_output = MagicMock()
        mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_sequential.return_value.return_value = mock_output
        
        # Mock the image loading
        mock_image = MagicMock()
        mock_image.unsqueeze.return_value = MagicMock()
        mock_load_image.return_value = mock_image
        
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Execute the operator
        result = operator.run({"path": self.test_image_path})
        
        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        
        # Check that the image was loaded
        mock_load_image.assert_called_once_with(self.test_image_path)
    
    @patch("operators.image_vec_rep_resnet.image_vec_rep_resnet_v2.ImageVecRepResNet._load_image")
    @patch("torch.nn.Sequential")
    @patch("torchvision.models.resnet50")
    def test_execute_failure(self, mock_resnet, mock_sequential, mock_load_image):
        """Test operator execution failure."""
        # Mock the model and its output
        mock_model = MagicMock()
        mock_model.children.return_value = [MagicMock()]
        mock_resnet.return_value = mock_model
        
        # Make the image loading fail
        mock_load_image.side_effect = ValueError("Image loading failed")
        
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Check that the operator raises an OperatorExecutionError
        with self.assertRaises(OperatorExecutionError):
            operator.run(self.test_image_path)
    
    def test_validate_input_with_invalid_path(self):
        """Test input validation with an invalid path."""
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Check that the operator rejects an invalid path
        self.assertFalse(operator._validate_input("invalid_path"))
    
    def test_validate_input_with_invalid_extension(self):
        """Test input validation with an invalid file extension."""
        # Create a temporary file with an invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Initialize the operator
            operator = ImageVecRepResNet()
            
            # Check that the operator rejects a file with an invalid extension
            self.assertFalse(operator._validate_input(temp_file.name))
    
    def test_is_valid_image_extension(self):
        """Test the _is_valid_image_extension method."""
        # Initialize the operator
        operator = ImageVecRepResNet()
        
        # Check valid extensions
        self.assertTrue(operator._is_valid_image_extension("image.jpg"))
        self.assertTrue(operator._is_valid_image_extension("image.jpeg"))
        self.assertTrue(operator._is_valid_image_extension("image.png"))
        self.assertTrue(operator._is_valid_image_extension("image.bmp"))
        self.assertTrue(operator._is_valid_image_extension("image.gif"))
        self.assertTrue(operator._is_valid_image_extension("image.webp"))
        
        # Check invalid extensions
        self.assertFalse(operator._is_valid_image_extension("image.txt"))
        self.assertFalse(operator._is_valid_image_extension("image.pdf"))
        self.assertFalse(operator._is_valid_image_extension("image"))


if __name__ == "__main__":
    unittest.main()
