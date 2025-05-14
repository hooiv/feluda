"""
Text Sentiment Analysis Operator

This module provides an operator for analyzing sentiment in text using a pre-trained model.
It demonstrates the use of the new BaseFeludaOperator class with formal verification hooks
and contract programming.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import deal
import numpy as np
from pydantic import BaseModel, Field, field_validator

from feluda.base_operator import BaseFeludaOperator
from feluda.exceptions import OperatorExecutionError, OperatorInitializationError
from feluda.models.data_models import MediaContent, OperatorResult

log = logging.getLogger(__name__)

# Try to import transformers, but don't fail if it's not available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    log.warning("Transformers library not available. Using fallback sentiment analysis.")


class TextSentimentAnalysisParameters(BaseModel):
    """Parameters for the text sentiment analysis operator."""
    
    model_name: str = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        description="The name of the pre-trained model to use for sentiment analysis."
    )
    
    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU for inference if available."
    )
    
    fallback_enabled: bool = Field(
        default=True,
        description="Whether to use a fallback method if transformers is not available."
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate the model name."""
        if not v:
            raise ValueError("Model name cannot be empty")
        return v


class TextSentimentAnalysis(BaseFeludaOperator[Union[str, Dict[str, Any], MediaContent], Dict[str, Any], TextSentimentAnalysisParameters]):
    """
    Operator for analyzing sentiment in text.
    
    This operator takes text as input and returns sentiment analysis results.
    """
    
    name = "TextSentimentAnalysis"
    description = "Analyze sentiment in text using a pre-trained model."
    version = "1.0.0"
    parameters_model = TextSentimentAnalysisParameters
    
    def _initialize(self) -> None:
        """
        Initialize the operator.
        
        This method loads the sentiment analysis model and prepares it for inference.
        
        Raises:
            OperatorInitializationError: If initialization fails.
        """
        try:
            self.model = None
            
            if TRANSFORMERS_AVAILABLE:
                # Initialize the transformers pipeline
                device = 0 if self.parameters.use_gpu else -1
                self.model = pipeline(
                    "sentiment-analysis",
                    model=self.parameters.model_name,
                    device=device
                )
                log.info(f"Initialized sentiment analysis model: {self.parameters.model_name}")
            elif not self.parameters.fallback_enabled:
                raise OperatorInitializationError(
                    message="Transformers library not available and fallback is disabled",
                    operator_type=self.__class__.__name__,
                    details={"model_name": self.parameters.model_name}
                )
            else:
                log.warning("Using fallback sentiment analysis method")
        
        except Exception as e:
            if not isinstance(e, OperatorInitializationError):
                raise OperatorInitializationError(
                    message=f"Failed to initialize sentiment analysis model: {str(e)}",
                    operator_type=self.__class__.__name__,
                    details={"model_name": self.parameters.model_name}
                ) from e
            raise
    
    def _validate_input(self, input_data: Union[str, Dict[str, Any], MediaContent]) -> bool:
        """
        Validate the input data.
        
        Args:
            input_data: The input data to validate.
            
        Returns:
            True if the input data is valid, False otherwise.
        """
        if isinstance(input_data, str):
            return bool(input_data.strip())
        
        elif isinstance(input_data, dict) and "text" in input_data:
            return bool(input_data["text"].strip())
        
        elif isinstance(input_data, MediaContent):
            if input_data.metadata.media_type != "text":
                return False
            
            if input_data.content_data:
                return bool(str(input_data.content_data).strip())
            
            return bool(input_data.content_uri)
        
        return False
    
    @deal.ensure(lambda result: isinstance(result, dict) and "sentiment" in result)
    def _execute(self, input_data: Union[str, Dict[str, Any], MediaContent]) -> Dict[str, Any]:
        """
        Execute the operator on the given input data.
        
        Args:
            input_data: The input data to process. Can be a string, a dictionary with a "text" key,
                       or a MediaContent object.
            
        Returns:
            A dictionary containing the sentiment analysis results.
            
        Raises:
            OperatorExecutionError: If execution fails.
        """
        try:
            # Extract the text from the input data
            if isinstance(input_data, str):
                text = input_data
            elif isinstance(input_data, dict) and "text" in input_data:
                text = input_data["text"]
            elif isinstance(input_data, MediaContent):
                if input_data.content_data:
                    text = str(input_data.content_data)
                elif input_data.content_uri:
                    # In a real implementation, we would fetch the text from the URI
                    # For simplicity, we'll just use a placeholder
                    text = "Sample text from URI"
                else:
                    raise OperatorExecutionError(
                        message="No text content available in MediaContent",
                        operator_type=self.__class__.__name__,
                        input_data=input_data,
                        cause=ValueError("No text content available")
                    )
            else:
                raise OperatorExecutionError(
                    message="Invalid input data type",
                    operator_type=self.__class__.__name__,
                    input_data=input_data,
                    cause=TypeError(f"Expected str, dict, or MediaContent, got {type(input_data)}")
                )
            
            # Analyze sentiment
            if self.model is not None:
                # Use the transformers pipeline
                result = self.model(text)
                
                # Extract the sentiment and score
                sentiment = result[0]["label"]
                score = result[0]["score"]
                
                return {
                    "sentiment": sentiment,
                    "score": score,
                    "text": text,
                    "method": "transformers",
                    "model": self.parameters.model_name
                }
            else:
                # Use a simple fallback method
                return self._fallback_sentiment_analysis(text)
        
        except Exception as e:
            if isinstance(e, OperatorExecutionError):
                raise
            
            raise OperatorExecutionError(
                message=f"Failed to analyze sentiment: {str(e)}",
                operator_type=self.__class__.__name__,
                input_data=input_data,
                cause=e
            ) from e
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform a simple fallback sentiment analysis.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary containing the sentiment analysis results.
        """
        # This is a very simple sentiment analysis based on keyword matching
        # In a real implementation, this would be more sophisticated
        
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "dislike"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "NEGATIVE"
            score = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "NEUTRAL"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "score": score,
            "text": text,
            "method": "fallback",
            "model": "keyword_matching"
        }


# For backward compatibility with the old operator interface
_operator_instance = None

def initialize(parameters: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize the operator.
    
    Args:
        parameters: Optional dictionary of parameters to configure the operator.
    """
    global _operator_instance
    _operator_instance = TextSentimentAnalysis(parameters=parameters)

def run(input_data: Union[str, Dict[str, Any], MediaContent]) -> Dict[str, Any]:
    """
    Run the operator.
    
    Args:
        input_data: The input data to process.
        
    Returns:
        The sentiment analysis results.
    """
    global _operator_instance
    if _operator_instance is None:
        initialize()
    
    return _operator_instance.run(input_data)
