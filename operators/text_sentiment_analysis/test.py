"""
Test module for the Text Sentiment Analysis operator.
"""

import unittest
from unittest import mock

from feluda.models.data_models import MediaContent, MediaMetadata, MediaType

import operators.text_sentiment_analysis.text_sentiment_analysis as text_sentiment_analysis


class TestTextSentimentAnalysis(unittest.TestCase):
    """Test cases for the Text Sentiment Analysis operator."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize the operator with default parameters
        text_sentiment_analysis.initialize({})
    
    def tearDown(self):
        """Clean up after the test."""
        # Reset the operator instance
        text_sentiment_analysis._operator_instance = None
    
    def test_initialization(self):
        """Test that the operator initializes correctly."""
        # Check that the operator instance was created
        self.assertIsNotNone(text_sentiment_analysis._operator_instance)
        
        # Check that the operator has the correct attributes
        operator = text_sentiment_analysis._operator_instance
        self.assertEqual(operator.name, "TextSentimentAnalysis")
        self.assertEqual(operator.version, "1.0.0")
        
        # Check that the parameters were set correctly
        self.assertEqual(
            operator.parameters.model_name,
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.assertFalse(operator.parameters.use_gpu)
        self.assertTrue(operator.parameters.fallback_enabled)
    
    def test_initialization_with_parameters(self):
        """Test that the operator initializes correctly with custom parameters."""
        # Reset the operator instance
        text_sentiment_analysis._operator_instance = None
        
        # Initialize with custom parameters
        text_sentiment_analysis.initialize({
            "model_name": "custom-model",
            "use_gpu": True,
            "fallback_enabled": False
        })
        
        # Check that the parameters were set correctly
        operator = text_sentiment_analysis._operator_instance
        self.assertEqual(operator.parameters.model_name, "custom-model")
        self.assertTrue(operator.parameters.use_gpu)
        self.assertFalse(operator.parameters.fallback_enabled)
    
    def test_run_with_string(self):
        """Test running the operator with a string input."""
        # Run the operator with a positive text
        result = text_sentiment_analysis.run("I love this product, it's amazing!")
        
        # Check the result
        self.assertIn("sentiment", result)
        self.assertIn("score", result)
        self.assertIn("text", result)
        self.assertIn("method", result)
        self.assertIn("model", result)
        
        # For the fallback method, this should be classified as positive
        if result["method"] == "fallback":
            self.assertEqual(result["sentiment"], "POSITIVE")
            self.assertGreater(result["score"], 0.5)
        
        # Run the operator with a negative text
        result = text_sentiment_analysis.run("I hate this product, it's terrible!")
        
        # Check the result
        self.assertIn("sentiment", result)
        self.assertIn("score", result)
        
        # For the fallback method, this should be classified as negative
        if result["method"] == "fallback":
            self.assertEqual(result["sentiment"], "NEGATIVE")
            self.assertGreater(result["score"], 0.5)
    
    def test_run_with_dict(self):
        """Test running the operator with a dictionary input."""
        # Run the operator with a dictionary containing text
        result = text_sentiment_analysis.run({"text": "This is a great day!"})
        
        # Check the result
        self.assertIn("sentiment", result)
        self.assertIn("score", result)
        self.assertEqual(result["text"], "This is a great day!")
    
    def test_run_with_media_content(self):
        """Test running the operator with a MediaContent input."""
        # Create a MediaContent object
        media_content = MediaContent(
            metadata=MediaMetadata(media_type=MediaType.TEXT),
            content_data="I'm feeling happy today!"
        )
        
        # Run the operator with the MediaContent object
        result = text_sentiment_analysis.run(media_content)
        
        # Check the result
        self.assertIn("sentiment", result)
        self.assertIn("score", result)
        self.assertEqual(result["text"], "I'm feeling happy today!")
    
    def test_invalid_input(self):
        """Test that the operator handles invalid input correctly."""
        # Test with an empty string
        with self.assertRaises(Exception):
            text_sentiment_analysis.run("")
        
        # Test with an empty dictionary
        with self.assertRaises(Exception):
            text_sentiment_analysis.run({})
        
        # Test with a dictionary without a text key
        with self.assertRaises(Exception):
            text_sentiment_analysis.run({"not_text": "This should fail"})
        
        # Test with a MediaContent object with the wrong media type
        media_content = MediaContent(
            metadata=MediaMetadata(media_type=MediaType.IMAGE),
            content_data="This should fail"
        )
        with self.assertRaises(Exception):
            text_sentiment_analysis.run(media_content)
    
    @mock.patch("operators.text_sentiment_analysis.text_sentiment_analysis.TRANSFORMERS_AVAILABLE", False)
    def test_fallback_method(self):
        """Test that the fallback method works correctly."""
        # Reset the operator instance to force reinitialization
        text_sentiment_analysis._operator_instance = None
        text_sentiment_analysis.initialize({})
        
        # Run the operator with a positive text
        result = text_sentiment_analysis.run("I love this product, it's amazing!")
        
        # Check that the fallback method was used
        self.assertEqual(result["method"], "fallback")
        self.assertEqual(result["model"], "keyword_matching")
        
        # Check the sentiment
        self.assertEqual(result["sentiment"], "POSITIVE")
        self.assertGreater(result["score"], 0.5)
    
    @mock.patch("operators.text_sentiment_analysis.text_sentiment_analysis.TRANSFORMERS_AVAILABLE", False)
    def test_fallback_disabled(self):
        """Test that the operator fails when fallback is disabled and transformers is not available."""
        # Reset the operator instance
        text_sentiment_analysis._operator_instance = None
        
        # Initialize with fallback disabled
        with self.assertRaises(Exception):
            text_sentiment_analysis.initialize({"fallback_enabled": False})
    
    @mock.patch("operators.text_sentiment_analysis.text_sentiment_analysis.pipeline")
    def test_transformers_method(self, mock_pipeline):
        """Test that the transformers method works correctly."""
        # Mock the pipeline to return a fixed result
        mock_pipeline.return_value = lambda text: [{"label": "POSITIVE", "score": 0.95}]
        
        # Reset the operator instance to force reinitialization
        text_sentiment_analysis._operator_instance = None
        text_sentiment_analysis.initialize({})
        
        # Run the operator
        result = text_sentiment_analysis.run("This is a test")
        
        # Check that the transformers method was used
        self.assertEqual(result["method"], "transformers")
        self.assertEqual(result["sentiment"], "POSITIVE")
        self.assertEqual(result["score"], 0.95)
        
        # Check that the pipeline was called with the correct arguments
        mock_pipeline.assert_called_once_with(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )


if __name__ == "__main__":
    unittest.main()
