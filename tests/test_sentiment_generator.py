"""
Test suite for sentiment-controlled text generation.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
import tempfile
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_generator import (
    SentimentControlledGenerator, 
    SentimentType, 
    GenerationConfig,
    create_synthetic_dataset
)
from config.config import AppConfig, ModelConfig, GenerationConfig as ConfigGenerationConfig


class TestSentimentType(unittest.TestCase):
    """Test SentimentType enum."""
    
    def test_sentiment_values(self):
        """Test sentiment enum values."""
        self.assertEqual(SentimentType.POSITIVE.value, "positive")
        self.assertEqual(SentimentType.NEGATIVE.value, "negative")
        self.assertEqual(SentimentType.NEUTRAL.value, "neutral")


class TestGenerationConfig(unittest.TestCase):
    """Test GenerationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        self.assertEqual(config.max_length, 100)
        self.assertEqual(config.num_return_sequences, 1)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertTrue(config.do_sample)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_length=50,
            temperature=1.0,
            num_return_sequences=3
        )
        self.assertEqual(config.max_length, 50)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.num_return_sequences, 3)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig class."""
    
    def test_default_config(self):
        """Test default application configuration."""
        config = AppConfig()
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.generation, ConfigGenerationConfig)
        self.assertEqual(config.logging_level, "INFO")
    
    def test_config_serialization(self):
        """Test configuration serialization to YAML."""
        config = AppConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config.to_yaml(config_path)
            
            # Load and verify
            loaded_config = AppConfig.from_yaml(config_path)
            self.assertEqual(config.model.name, loaded_config.model.name)
            self.assertEqual(config.generation.max_length, loaded_config.generation.max_length)
            
        finally:
            os.unlink(config_path)


class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset creation."""
    
    def test_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_samples=10)
        
        self.assertEqual(len(dataset), 10)
        self.assertIn('prompt', dataset.column_names)
        self.assertIn('sentiment', dataset.column_names)
        self.assertIn('expected_keywords', dataset.column_names)
        
        # Check sentiment values
        sentiments = dataset['sentiment']
        valid_sentiments = {'positive', 'negative', 'neutral'}
        self.assertTrue(all(s in valid_sentiments for s in sentiments))


class TestSentimentControlledGenerator(unittest.TestCase):
    """Test SentimentControlledGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid downloading large models during tests
        self.mock_generator = Mock()
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
    
    @patch('sentiment_generator.AutoTokenizer')
    @patch('sentiment_generator.AutoModelForCausalLM')
    @patch('sentiment_generator.pipeline')
    def test_generator_initialization(self, mock_pipeline, mock_model_class, mock_tokenizer_class):
        """Test generator initialization."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        mock_pipeline.return_value = self.mock_generator
        
        # Initialize generator
        generator = SentimentControlledGenerator(model_name="distilgpt2")
        
        # Verify initialization
        self.assertEqual(generator.model_name, "distilgpt2")
        self.assertIsNotNone(generator.tokenizer)
        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.generator)
    
    def test_create_sentiment_prompt(self):
        """Test sentiment prompt creation."""
        generator = SentimentControlledGenerator.__new__(SentimentControlledGenerator)
        
        # Test positive sentiment prompt
        positive_prompt = generator._create_sentiment_prompt("test", SentimentType.POSITIVE)
        self.assertIn("positive", positive_prompt.lower())
        self.assertIn("optimistic", positive_prompt.lower())
        
        # Test negative sentiment prompt
        negative_prompt = generator._create_sentiment_prompt("test", SentimentType.NEGATIVE)
        self.assertIn("negative", negative_prompt.lower())
        self.assertIn("pessimistic", negative_prompt.lower())
        
        # Test neutral sentiment prompt
        neutral_prompt = generator._create_sentiment_prompt("test", SentimentType.NEUTRAL)
        self.assertIn("neutral", neutral_prompt.lower())
        self.assertIn("objective", neutral_prompt.lower())
    
    def test_batch_generate_validation(self):
        """Test batch generation input validation."""
        generator = SentimentControlledGenerator.__new__(SentimentControlledGenerator)
        
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            generator.batch_generate(
                ["prompt1", "prompt2"],
                [SentimentType.POSITIVE]  # Only one sentiment
            )


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with mocked components."""
        with patch('sentiment_generator.AutoTokenizer') as mock_tokenizer_class, \
             patch('sentiment_generator.AutoModelForCausalLM') as mock_model_class, \
             patch('sentiment_generator.pipeline') as mock_pipeline:
            
            # Setup mocks
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.pad_token_id = 50256
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            mock_gen_pipeline = Mock()
            mock_gen_pipeline.return_value = [{'generated_text': 'This is a positive test sentence.'}]
            mock_pipeline.return_value = mock_gen_pipeline
            
            # Create generator
            generator = SentimentControlledGenerator(model_name="distilgpt2")
            
            # Test generation
            result = generator.generate_with_sentiment(
                "The weather is",
                SentimentType.POSITIVE
            )
            
            # Verify result structure
            self.assertIsNotNone(result.generated_text)
            self.assertEqual(result.sentiment, SentimentType.POSITIVE)
            self.assertIsInstance(result.confidence, float)
            self.assertEqual(result.prompt, "The weather is")
            self.assertEqual(result.model_name, "distilgpt2")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSentimentType,
        TestGenerationConfig,
        TestAppConfig,
        TestSyntheticDataset,
        TestSentimentControlledGenerator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
