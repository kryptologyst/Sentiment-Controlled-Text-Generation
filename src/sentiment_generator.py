"""
Sentiment-Controlled Text Generation Module

This module provides functionality for generating text with controlled sentiment
using state-of-the-art transformer models and techniques.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    TextGenerationPipeline
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Enumeration for sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    pad_token_id: Optional[int] = None


@dataclass
class SentimentResult:
    """Result container for sentiment-controlled generation."""
    generated_text: str
    sentiment: SentimentType
    confidence: float
    prompt: str
    model_name: str


class SentimentControlledGenerator:
    """
    A class for generating text with controlled sentiment using transformer models.
    
    This class supports multiple approaches:
    1. Prompt-based sentiment control
    2. Fine-tuned models for specific sentiments
    3. Zero-shot sentiment classification
    """
    
    def __init__(
        self, 
        model_name: str = "gpt2",
        device: Optional[str] = None,
        use_accelerate: bool = True
    ):
        """
        Initialize the sentiment-controlled generator.
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            use_accelerate: Whether to use accelerate for optimization
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_accelerate = use_accelerate
        
        # Initialize components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generator: Optional[TextGenerationPipeline] = None
        self.sentiment_classifier: Optional[TextGenerationPipeline] = None
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Initialized SentimentControlledGenerator with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.use_accelerate else None
            )
            
            if not self.use_accelerate:
                self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_with_sentiment(
        self,
        prompt: str,
        sentiment: SentimentType,
        config: Optional[GenerationConfig] = None
    ) -> SentimentResult:
        """
        Generate text with controlled sentiment.
        
        Args:
            prompt: Input prompt for generation
            sentiment: Desired sentiment for the generated text
            config: Generation configuration
            
        Returns:
            SentimentResult containing generated text and metadata
        """
        if config is None:
            config = GenerationConfig()
        
        # Create sentiment-aware prompt
        sentiment_prompt = self._create_sentiment_prompt(prompt, sentiment)
        
        try:
            # Generate text
            results = self.generator(
                sentiment_prompt,
                max_length=config.max_length,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            generated_text = results[0]['generated_text']
            
            # Remove the original prompt from generated text
            if generated_text.startswith(sentiment_prompt):
                generated_text = generated_text[len(sentiment_prompt):].strip()
            
            # Verify sentiment
            confidence = self._verify_sentiment(generated_text, sentiment)
            
            return SentimentResult(
                generated_text=generated_text,
                sentiment=sentiment,
                confidence=confidence,
                prompt=prompt,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def _create_sentiment_prompt(self, prompt: str, sentiment: SentimentType) -> str:
        """
        Create a sentiment-aware prompt.
        
        Args:
            prompt: Original prompt
            sentiment: Desired sentiment
            
        Returns:
            Enhanced prompt with sentiment guidance
        """
        sentiment_guidance = {
            SentimentType.POSITIVE: "Write in a positive, optimistic, and uplifting tone:",
            SentimentType.NEGATIVE: "Write in a negative, pessimistic, and critical tone:",
            SentimentType.NEUTRAL: "Write in a neutral, objective, and balanced tone:"
        }
        
        return f"{sentiment_guidance[sentiment]} {prompt}"
    
    def _verify_sentiment(self, text: str, expected_sentiment: SentimentType) -> float:
        """
        Verify the sentiment of generated text using zero-shot classification.
        
        Args:
            text: Generated text to analyze
            expected_sentiment: Expected sentiment
            
        Returns:
            Confidence score for the sentiment match
        """
        try:
            # Use a simple sentiment analysis pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            result = sentiment_pipeline(text)[0]
            
            # Map sentiment labels to our enum
            label_mapping = {
                'LABEL_0': SentimentType.NEGATIVE,
                'LABEL_1': SentimentType.NEUTRAL,
                'LABEL_2': SentimentType.POSITIVE
            }
            
            predicted_sentiment = label_mapping.get(result['label'], SentimentType.NEUTRAL)
            
            # Return confidence if sentiment matches, otherwise return 1 - confidence
            if predicted_sentiment == expected_sentiment:
                return result['score']
            else:
                return 1 - result['score']
                
        except Exception as e:
            logger.warning(f"Sentiment verification failed: {e}")
            return 0.5  # Return neutral confidence if verification fails
    
    def batch_generate(
        self,
        prompts: List[str],
        sentiments: List[SentimentType],
        config: Optional[GenerationConfig] = None
    ) -> List[SentimentResult]:
        """
        Generate text for multiple prompts and sentiments.
        
        Args:
            prompts: List of input prompts
            sentiments: List of desired sentiments
            config: Generation configuration
            
        Returns:
            List of SentimentResult objects
        """
        if len(prompts) != len(sentiments):
            raise ValueError("Number of prompts must match number of sentiments")
        
        results = []
        for prompt, sentiment in zip(prompts, sentiments):
            result = self.generate_with_sentiment(prompt, sentiment, config)
            results.append(result)
        
        return results
    
    def evaluate_sentiment_control(
        self,
        test_prompts: List[str],
        expected_sentiments: List[SentimentType],
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, float]:
        """
        Evaluate the effectiveness of sentiment control.
        
        Args:
            test_prompts: List of test prompts
            expected_sentiments: List of expected sentiments
            config: Generation configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.batch_generate(test_prompts, expected_sentiments, config)
        
        # Calculate accuracy
        correct_predictions = sum(
            1 for result in results 
            if result.confidence > 0.5
        )
        accuracy = correct_predictions / len(results)
        
        # Calculate average confidence
        avg_confidence = np.mean([result.confidence for result in results])
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'total_samples': len(results)
        }
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
            'max_length': self.tokenizer.model_max_length if self.tokenizer else 0
        }


def create_synthetic_dataset(num_samples: int = 100) -> Dataset:
    """
    Create a synthetic dataset for testing sentiment-controlled generation.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Hugging Face Dataset object
    """
    import random
    
    # Base prompts for different topics
    base_prompts = [
        "The weather today is",
        "I feel about my job",
        "The movie I watched was",
        "My relationship with my family is",
        "The food at the restaurant was",
        "My experience with technology has been",
        "The book I'm reading is",
        "My thoughts on the economy are",
        "The vacation I took was",
        "My opinion on social media is"
    ]
    
    sentiments = [SentimentType.POSITIVE, SentimentType.NEGATIVE, SentimentType.NEUTRAL]
    
    data = {
        'prompt': [],
        'sentiment': [],
        'expected_keywords': []
    }
    
    # Define expected keywords for each sentiment
    sentiment_keywords = {
        SentimentType.POSITIVE: ['great', 'amazing', 'wonderful', 'excellent', 'fantastic', 'love', 'enjoy'],
        SentimentType.NEGATIVE: ['terrible', 'awful', 'horrible', 'disappointing', 'hate', 'dislike', 'bad'],
        SentimentType.NEUTRAL: ['okay', 'average', 'normal', 'fine', 'decent', 'acceptable', 'standard']
    }
    
    for _ in range(num_samples):
        prompt = random.choice(base_prompts)
        sentiment = random.choice(sentiments)
        
        data['prompt'].append(prompt)
        data['sentiment'].append(sentiment.value)
        data['expected_keywords'].append(sentiment_keywords[sentiment])
    
    return Dataset.from_dict(data)


if __name__ == "__main__":
    # Example usage
    generator = SentimentControlledGenerator()
    
    # Test single generation
    result = generator.generate_with_sentiment(
        "The weather today is",
        SentimentType.POSITIVE
    )
    
    print(f"Generated: {result.generated_text}")
    print(f"Sentiment: {result.sentiment.value}")
    print(f"Confidence: {result.confidence:.2f}")
