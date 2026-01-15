"""
Data generation and management utilities.
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from datasets import Dataset
import pandas as pd


def create_sample_prompts() -> List[str]:
    """Create a collection of sample prompts for testing."""
    prompts = [
        "The weather today is",
        "I feel about my job",
        "The movie I watched was",
        "My relationship with my family is",
        "The food at the restaurant was",
        "My experience with technology has been",
        "The book I'm reading is",
        "My thoughts on the economy are",
        "The vacation I took was",
        "My opinion on social media is",
        "The new product launch was",
        "My experience with customer service was",
        "The conference I attended was",
        "My feelings about the future are",
        "The project I worked on was",
        "My opinion on climate change is",
        "The restaurant I visited was",
        "My experience with online shopping was",
        "The concert I went to was",
        "My thoughts on artificial intelligence are"
    ]
    return prompts


def create_sentiment_dataset(
    num_samples: int = 100,
    output_path: Optional[str] = None
) -> Dataset:
    """
    Create a comprehensive sentiment dataset for training and evaluation.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Optional path to save the dataset
        
    Returns:
        Hugging Face Dataset object
    """
    prompts = create_sample_prompts()
    
    # Define sentiment templates and keywords
    sentiment_templates = {
        'positive': [
            "amazing", "wonderful", "fantastic", "excellent", "outstanding",
            "incredible", "brilliant", "perfect", "delightful", "spectacular"
        ],
        'negative': [
            "terrible", "awful", "horrible", "disappointing", "frustrating",
            "annoying", "disgusting", "pathetic", "miserable", "atrocious"
        ],
        'neutral': [
            "average", "okay", "decent", "acceptable", "standard",
            "normal", "typical", "ordinary", "regular", "moderate"
        ]
    }
    
    data = {
        'prompt': [],
        'sentiment': [],
        'expected_keywords': [],
        'context': [],
        'difficulty': []
    }
    
    for _ in range(num_samples):
        prompt = random.choice(prompts)
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        
        # Add context based on sentiment
        contexts = {
            'positive': [
                "This is a great experience that I would recommend to others.",
                "I'm really happy with this and feel optimistic about the future.",
                "This exceeded my expectations and made my day better."
            ],
            'negative': [
                "This was a disappointing experience that I wouldn't recommend.",
                "I'm frustrated with this and feel pessimistic about similar situations.",
                "This fell short of my expectations and made me feel worse."
            ],
            'neutral': [
                "This was an average experience with no strong feelings either way.",
                "I have mixed feelings about this and remain neutral.",
                "This met my basic expectations without being remarkable."
            ]
        }
        
        context = random.choice(contexts[sentiment])
        difficulty = random.choice(['easy', 'medium', 'hard'])
        
        data['prompt'].append(prompt)
        data['sentiment'].append(sentiment)
        data['expected_keywords'].append(sentiment_templates[sentiment])
        data['context'].append(context)
        data['difficulty'].append(difficulty)
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    
    # Save if output path provided
    if output_path:
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    
    return dataset


def create_evaluation_dataset() -> Dataset:
    """Create a smaller dataset specifically for evaluation."""
    evaluation_prompts = [
        "The service at the hotel was",
        "My experience with the new app is",
        "The quality of the product is",
        "I think the new policy is",
        "The performance of the team was",
        "My opinion on the recent changes is",
        "The customer support I received was",
        "I feel about the current situation is",
        "The results of the experiment were",
        "My reaction to the news was"
    ]
    
    data = {
        'prompt': [],
        'sentiment': [],
        'expected_keywords': [],
        'context': []
    }
    
    sentiments = ['positive', 'negative', 'neutral']
    
    for prompt in evaluation_prompts:
        for sentiment in sentiments:
            data['prompt'].append(prompt)
            data['sentiment'].append(sentiment)
            
            # Add appropriate keywords
            keywords = {
                'positive': ['excellent', 'great', 'wonderful', 'amazing'],
                'negative': ['terrible', 'awful', 'disappointing', 'bad'],
                'neutral': ['okay', 'average', 'decent', 'normal']
            }
            data['expected_keywords'].append(keywords[sentiment])
            
            # Add context
            contexts = {
                'positive': f"The {prompt.lower()} truly exceptional and exceeded expectations.",
                'negative': f"The {prompt.lower()} quite disappointing and below expectations.",
                'neutral': f"The {prompt.lower()} average and met basic expectations."
            }
            data['context'].append(contexts[sentiment])
    
    return Dataset.from_dict(data)


def save_prompts_to_file(prompts: List[str], file_path: str):
    """Save prompts to a text file for batch processing."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    print(f"Prompts saved to {file_path}")


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def create_config_files():
    """Create default configuration files."""
    # Create default YAML config
    default_config = {
        'model': {
            'name': 'gpt2',
            'device': 'auto',
            'use_accelerate': True,
            'torch_dtype': 'auto'
        },
        'generation': {
            'max_length': 100,
            'num_return_sequences': 1,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'repetition_penalty': 1.1
        },
        'ui': {
            'title': 'Sentiment-Controlled Text Generation',
            'description': 'Generate text with controlled sentiment using AI',
            'max_prompt_length': 500,
            'default_num_samples': 3
        },
        'logging_level': 'INFO',
        'cache_dir': './cache'
    }
    
    # Save YAML config
    import yaml
    with open('config/default_config.yaml', 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    # Save JSON config
    with open('config/default_config.json', 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print("Configuration files created successfully!")


def main():
    """Main function for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets and sample data")
    parser.add_argument('--dataset-size', type=int, default=100, help='Size of the dataset')
    parser.add_argument('--output-dir', default='./data', help='Output directory')
    parser.add_argument('--create-configs', action='store_true', help='Create default config files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_sentiment_dataset(
        num_samples=args.dataset_size,
        output_path=os.path.join(args.output_dir, 'train_dataset')
    )
    
    print("Creating evaluation dataset...")
    eval_dataset = create_evaluation_dataset()
    eval_dataset.save_to_disk(os.path.join(args.output_dir, 'eval_dataset'))
    
    # Create sample prompts file
    sample_prompts = create_sample_prompts()
    save_prompts_to_file(sample_prompts, os.path.join(args.output_dir, 'sample_prompts.txt'))
    
    # Create config files if requested
    if args.create_configs:
        create_config_files()
    
    print(f"Data generation completed! Files saved to {args.output_dir}")
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Evaluation dataset: {len(eval_dataset)} samples")
    print(f"Sample prompts: {len(sample_prompts)} prompts")


if __name__ == "__main__":
    main()
