"""
Image Vector Representation using ResNet - Version 2

This module provides an operator for generating vector representations (embeddings)
of images using ResNet models. This is a refactored version using the new BaseFeludaOperator
class with formal verification hooks and contract programming.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import deal
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from torchvision import models, transforms

from feluda.base_operator import BaseFeludaOperator
from feluda.exceptions import OperatorExecutionError, OperatorInitializationError
from feluda.models.data_models import MediaContent

log = logging.getLogger(__name__)


class ImageVecRepResNetParameters(BaseModel):
    """Parameters for the ImageVecRepResNet operator."""

    model_name: str = Field(
        default="resnet50",
        description="The name of the ResNet model to use (resnet18, resnet34, resnet50, resnet101, resnet152)."
    )

    use_pretrained: bool = Field(
        default=True,
        description="Whether to use pretrained weights."
    )

    device: str = Field(
        default="auto",
        description="Device to use for inference ('cpu', 'cuda', or 'auto' to automatically select)."
    )

    batch_size: int = Field(
        default=32,
        description="Batch size for processing multiple images."
    )

    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize the embeddings."
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that the model name is supported."""
        supported_models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        if v not in supported_models:
            raise ValueError(f"Model name '{v}' is not supported. Supported models: {supported_models}")
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and normalize the device specification."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if v not in ["cpu", "cuda"]:
            raise ValueError(f"Device '{v}' is not supported. Supported devices: 'cpu', 'cuda', 'auto'")
        if v == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return v


class ImageVecRepResNet(BaseFeludaOperator[Union[str, Dict[str, Any], MediaContent], np.ndarray, ImageVecRepResNetParameters]):
    """
    Operator for generating vector representations (embeddings) of images using ResNet models.

    This operator takes an image as input and returns a vector representation (embedding)
    of the image using a ResNet model.
    """

    name = "ImageVecRepResNet"
    description = "Generate vector representations (embeddings) of images using ResNet models."
    version = "2.0.0"
    parameters_model = ImageVecRepResNetParameters

    def _initialize(self) -> None:
        """
        Initialize the operator.

        This method loads the ResNet model and prepares it for inference.

        Raises:
            OperatorInitializationError: If initialization fails.
        """
        try:
            # Select the appropriate model
            model_func = getattr(models, self.parameters.model_name)
            self.model = model_func(pretrained=self.parameters.use_pretrained)

            # Remove the classification layer to get embeddings
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

            # Set the model to evaluation mode
            self.model.eval()

            # Move the model to the specified device
            self.device = torch.device(self.parameters.device)
            self.model = self.model.to(self.device)

            # Define the image transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            log.info(f"Initialized {self.name} with {self.parameters.model_name} on {self.device}")

        except Exception as e:
            raise OperatorInitializationError(
                message=f"Failed to initialize {self.name}",
                operator_type=self.name,
                details={"cause": str(e)}
            ) from e

    @deal.pre(lambda self, input_data: self._validate_input(input_data))
    @deal.post(lambda result: result is not None and isinstance(result, np.ndarray))
    @deal.post(lambda result: result.ndim == 1)
    @deal.raises(OperatorExecutionError)
    def _execute(self, input_data: Union[str, Dict[str, Any], MediaContent]) -> np.ndarray:
        """
        Execute the operator on the given input data.

        Args:
            input_data: The input data, which can be:
                - A string path to an image file
                - A dictionary with a 'path' key pointing to an image file
                - A MediaContent object

        Returns:
            A numpy array containing the image embedding.

        Raises:
            OperatorExecutionError: If execution fails.
        """
        try:
            # Process the input data to get the image path
            image_path = self._get_image_path(input_data)

            # Load and preprocess the image
            image = self._load_image(image_path)

            # Generate the embedding
            with torch.no_grad():
                image_tensor = image.unsqueeze(0).to(self.device)
                embedding = self.model(image_tensor).squeeze().cpu().numpy()

            # Normalize the embedding if requested
            if self.parameters.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            raise OperatorExecutionError(
                message=f"Failed to generate embedding for image",
                operator_type=self.name,
                input_data=input_data,
                cause=e
            ) from e

    def _validate_input(self, input_data: Union[str, Dict[str, Any], MediaContent]) -> bool:
        """
        Validate the input data.

        Args:
            input_data: The input data to validate.

        Returns:
            True if the input is valid, False otherwise.
        """
        if isinstance(input_data, str):
            return os.path.isfile(input_data) and self._is_valid_image_extension(input_data)

        if isinstance(input_data, dict):
            return "path" in input_data and os.path.isfile(input_data["path"]) and self._is_valid_image_extension(input_data["path"])

        if isinstance(input_data, MediaContent):
            if input_data.content_uri:
                return os.path.isfile(input_data.content_uri) and self._is_valid_image_extension(input_data.content_uri)
            return input_data.content_data is not None

        return False

    def _get_image_path(self, input_data: Union[str, Dict[str, Any], MediaContent]) -> str:
        """
        Get the image path from the input data.

        Args:
            input_data: The input data.

        Returns:
            The path to the image file.

        Raises:
            ValueError: If the image path cannot be determined.
        """
        if isinstance(input_data, str):
            return input_data

        if isinstance(input_data, dict):
            return input_data["path"]

        if isinstance(input_data, MediaContent):
            if input_data.content_uri:
                return input_data.content_uri

            # If we have content_data but no content_uri, we would need to save it temporarily
            # This is not implemented in this example
            raise ValueError("MediaContent with only content_data is not supported yet")

        raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.

        Args:
            image_path: The path to the image file.

        Returns:
            A preprocessed image tensor.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            return self.transform(image)

        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}") from e

    @staticmethod
    def _is_valid_image_extension(path: str) -> bool:
        """
        Check if the file has a valid image extension.

        Args:
            path: The path to check.

        Returns:
            True if the file has a valid image extension, False otherwise.
        """
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        return any(path.lower().endswith(ext) for ext in valid_extensions)


# For backward compatibility with the existing operator interface
def initialize(parameters: Dict[str, Any]) -> None:
    """
    Initialize the operator.

    Args:
        parameters: Parameters for initialization.
    """
    global operator
    operator = ImageVecRepResNet(parameters=parameters)


def run(input_data: Union[str, Dict[str, Any]]) -> np.ndarray:
    """
    Run the operator.

    Args:
        input_data: The input data.

    Returns:
        The embedding vector.
    """
    global operator
    return operator._execute(input_data)
