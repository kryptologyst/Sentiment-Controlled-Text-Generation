#!/usr/bin/env python3
"""
Command-line interface for sentiment-controlled text generation.
"""

import argparse
import sys
import os
from typing import List, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_generator import SentimentControlledGenerator, SentimentType, GenerationConfig
from config.config import AppConfig, load_config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with controlled sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "The weather today is" --sentiment positive
  %(prog)s "I feel about my job" --sentiment negative --samples 3
  %(prog)s --batch prompts.txt --output results.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'prompt',
        nargs='?',
        help='Text prompt for generation'
    )
    input_group.add_argument(
        '--batch',
        help='Path to file containing prompts (one per line)'
    )
    
    # Sentiment options
    parser.add_argument(
        '--sentiment',
        choices=['positive', 'negative', 'neutral'],
        default='positive',
        help='Desired sentiment for generated text'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum length of generated text'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.1-2.0)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        default='gpt2',
        help='Model name to use for generation'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to run the model on'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        help='Output file path (JSON format)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def save_results_to_file(results: List[dict], file_path: str):
    """Save results to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving results to '{file_path}': {e}")


def print_result(result, index: Optional[int] = None):
    """Print a single generation result."""
    prefix = f"[{index}] " if index is not None else ""
    print(f"{prefix}Generated Text: {result.generated_text}")
    print(f"{prefix}Sentiment: {result.sentiment.value}")
    print(f"{prefix}Confidence: {result.confidence:.2f}")
    print(f"{prefix}Model: {result.model_name}")
    print("-" * 50)


def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.model != 'gpt2':
        config.model.name = args.model
    if args.device != 'auto':
        config.model.device = args.device
    
    # Initialize generator
    try:
        print("Loading model...")
        generator = SentimentControlledGenerator(
            model_name=config.model.name,
            device=config.model.device,
            use_accelerate=config.model.use_accelerate
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Prepare generation configuration
    gen_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.samples
    )
    
    # Convert sentiment string to enum
    sentiment_map = {
        'positive': SentimentType.POSITIVE,
        'negative': SentimentType.NEGATIVE,
        'neutral': SentimentType.NEUTRAL
    }
    sentiment = sentiment_map[args.sentiment]
    
    # Process prompts
    if args.batch:
        prompts = load_prompts_from_file(args.batch)
        print(f"Processing {len(prompts)} prompts...")
        
        all_results = []
        for i, prompt in enumerate(prompts, 1):
            if args.verbose:
                print(f"\nProcessing prompt {i}/{len(prompts)}: {prompt}")
            
            try:
                results = generator.batch_generate(
                    [prompt] * args.samples,
                    [sentiment] * args.samples,
                    gen_config
                )
                all_results.extend(results)
                
                if args.verbose:
                    for j, result in enumerate(results):
                        print_result(result, j + 1)
                
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
                continue
        
        # Save results if output file specified
        if args.output:
            results_data = [
                {
                    'prompt': r.prompt,
                    'generated_text': r.generated_text,
                    'sentiment': r.sentiment.value,
                    'confidence': r.confidence,
                    'model_name': r.model_name
                }
                for r in all_results
            ]
            save_results_to_file(results_data, args.output)
        
        print(f"\nCompleted! Generated {len(all_results)} samples.")
        
    else:
        # Single prompt
        prompt = args.prompt
        print(f"Generating text for prompt: '{prompt}'")
        print(f"Desired sentiment: {args.sentiment}")
        print(f"Number of samples: {args.samples}")
        print("-" * 50)
        
        try:
            results = generator.batch_generate(
                [prompt] * args.samples,
                [sentiment] * args.samples,
                gen_config
            )
            
            for i, result in enumerate(results, 1):
                print_result(result, i)
            
            # Save results if output file specified
            if args.output:
                results_data = [
                    {
                        'prompt': r.prompt,
                        'generated_text': r.generated_text,
                        'sentiment': r.sentiment.value,
                        'confidence': r.confidence,
                        'model_name': r.model_name
                    }
                    for r in results
                ]
                save_results_to_file(results_data, args.output)
            
        except Exception as e:
            print(f"Error generating text: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
