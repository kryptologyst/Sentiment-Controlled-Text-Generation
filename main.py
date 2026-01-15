"""
Main entry point for the sentiment-controlled text generation application.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from sentiment_generator import SentimentControlledGenerator, SentimentType, GenerationConfig


def main():
    """Main function demonstrating the sentiment-controlled text generation."""
    print("🎭 Sentiment-Controlled Text Generation Demo")
    print("=" * 50)
    
    try:
        # Initialize generator
        print("Loading model...")
        generator = SentimentControlledGenerator(model_name="distilgpt2")
        print("✅ Model loaded successfully!")
        
        # Example prompts
        prompts = [
            "The weather today is",
            "I feel about my job",
            "The movie I watched was"
        ]
        
        sentiments = [SentimentType.POSITIVE, SentimentType.NEGATIVE, SentimentType.NEUTRAL]
        
        print("\n🚀 Generating sentiment-controlled text...")
        print("-" * 50)
        
        # Generate text for each prompt and sentiment combination
        for i, prompt in enumerate(prompts):
            sentiment = sentiments[i % len(sentiments)]
            
            print(f"\n📝 Prompt: '{prompt}'")
            print(f"🎯 Sentiment: {sentiment.value.upper()}")
            
            try:
                result = generator.generate_with_sentiment(prompt, sentiment)
                
                print(f"✨ Generated: {result.generated_text}")
                print(f"📊 Confidence: {result.confidence:.2f}")
                print(f"🤖 Model: {result.model_name}")
                
            except Exception as e:
                print(f"❌ Error generating text: {e}")
            
            print("-" * 30)
        
        # Show model information
        print("\n📋 Model Information:")
        model_info = generator.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
