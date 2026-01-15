#!/usr/bin/env python3
"""
Demo script for sentiment-controlled text generation.
This script demonstrates the key features of the modernized project.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def demo_basic_generation():
    """Demonstrate basic sentiment-controlled generation."""
    print("🎭 Basic Sentiment-Controlled Generation Demo")
    print("=" * 60)
    
    try:
        from sentiment_generator import SentimentControlledGenerator, SentimentType
        
        # Initialize with a smaller model for demo
        print("Loading DistilGPT-2 model...")
        generator = SentimentControlledGenerator(model_name="distilgpt2")
        print("✅ Model loaded successfully!")
        
        # Demo prompts
        demo_cases = [
            ("The weather today is", SentimentType.POSITIVE),
            ("I feel about my job", SentimentType.NEGATIVE),
            ("The movie I watched was", SentimentType.NEUTRAL)
        ]
        
        print("\n🚀 Generating sentiment-controlled text...")
        print("-" * 60)
        
        for i, (prompt, sentiment) in enumerate(demo_cases, 1):
            print(f"\n📝 Example {i}:")
            print(f"   Prompt: '{prompt}'")
            print(f"   Target Sentiment: {sentiment.value.upper()}")
            
            try:
                result = generator.generate_with_sentiment(prompt, sentiment)
                
                print(f"   ✨ Generated: {result.generated_text}")
                print(f"   📊 Confidence: {result.confidence:.2f}")
                
                # Interpret confidence
                if result.confidence > 0.7:
                    print(f"   🎯 High confidence sentiment match!")
                elif result.confidence > 0.5:
                    print(f"   ⚠️  Moderate confidence sentiment match")
                else:
                    print(f"   ❌ Low confidence - sentiment may not match target")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print("-" * 40)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n\n🔄 Batch Processing Demo")
    print("=" * 60)
    
    try:
        from sentiment_generator import SentimentControlledGenerator, SentimentType
        
        generator = SentimentControlledGenerator(model_name="distilgpt2")
        
        # Batch prompts
        prompts = [
            "The service was",
            "My experience with the app is",
            "The quality of the product is"
        ]
        
        sentiments = [
            SentimentType.POSITIVE,
            SentimentType.NEGATIVE,
            SentimentType.NEUTRAL
        ]
        
        print("Processing batch of prompts...")
        results = generator.batch_generate(prompts, sentiments)
        
        print(f"✅ Generated {len(results)} samples successfully!")
        
        # Show summary
        avg_confidence = sum(r.confidence for r in results) / len(results)
        print(f"📊 Average confidence: {avg_confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False


def demo_evaluation():
    """Demonstrate evaluation capabilities."""
    print("\n\n📊 Evaluation Demo")
    print("=" * 60)
    
    try:
        from sentiment_generator import SentimentControlledGenerator, SentimentType
        
        generator = SentimentControlledGenerator(model_name="distilgpt2")
        
        # Test prompts
        test_prompts = [
            "The weather is",
            "I feel about work",
            "The movie was"
        ]
        
        expected_sentiments = [
            SentimentType.POSITIVE,
            SentimentType.NEGATIVE,
            SentimentType.NEUTRAL
        ]
        
        print("Evaluating sentiment control effectiveness...")
        metrics = generator.evaluate_sentiment_control(
            test_prompts, 
            expected_sentiments
        )
        
        print("📈 Evaluation Results:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}")
        print(f"   Average Confidence: {metrics['average_confidence']:.2f}")
        print(f"   Total Samples: {metrics['total_samples']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False


def show_project_info():
    """Show project information and usage instructions."""
    print("\n\n📋 Project Information")
    print("=" * 60)
    
    print("🎯 Sentiment-Controlled Text Generation")
    print("   A modern implementation using transformer models")
    print()
    print("🚀 Quick Start Options:")
    print("   1. Web Interface: streamlit run web_app/app.py")
    print("   2. CLI Tool: python src/cli.py 'prompt' --sentiment positive")
    print("   3. Python API: See examples in README.md")
    print()
    print("📁 Project Structure:")
    print("   src/           - Core generation logic")
    print("   web_app/       - Streamlit web interface")
    print("   config/        - Configuration files")
    print("   data/          - Sample data and utilities")
    print("   tests/         - Test suite")
    print()
    print("🔧 Features:")
    print("   ✅ Sentiment control (positive/negative/neutral)")
    print("   ✅ Multiple model support (GPT-2, DistilGPT-2)")
    print("   ✅ Web interface with visualizations")
    print("   ✅ CLI for batch processing")
    print("   ✅ Comprehensive testing")
    print("   ✅ Configuration management")
    print("   ✅ Evaluation metrics")


def main():
    """Main demo function."""
    print("🎭 Sentiment-Controlled Text Generation")
    print("   Modernized Project Demo")
    print("=" * 60)
    
    success_count = 0
    total_demos = 3
    
    # Run demos
    if demo_basic_generation():
        success_count += 1
    
    if demo_batch_processing():
        success_count += 1
    
    if demo_evaluation():
        success_count += 1
    
    # Show project info
    show_project_info()
    
    # Summary
    print(f"\n🎉 Demo Summary: {success_count}/{total_demos} demos completed successfully!")
    
    if success_count == total_demos:
        print("✅ All demos passed! The project is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Try the web interface: streamlit run web_app/app.py")
        print("   2. Run tests: python -m pytest tests/")
        print("   3. Generate sample data: python data/generate_data.py")
    else:
        print("⚠️  Some demos failed. Check error messages above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return 0 if success_count == total_demos else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
