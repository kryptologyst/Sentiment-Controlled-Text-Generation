# Sentiment-Controlled Text Generation

A production-ready implementation of sentiment-controlled text generation using state-of-the-art transformer models. This project allows you to generate text with specific sentiment (positive, negative, or neutral) using advanced AI techniques.

## Features

- **Sentiment Control**: Generate text with precise sentiment control using prompt engineering
- **Multiple Models**: Support for GPT-2, DistilGPT-2, and other transformer models
- **Web Interface**: Beautiful Streamlit-based web application for easy interaction
- **CLI Interface**: Command-line tool for batch processing and automation
- **Analytics**: Built-in sentiment analysis and confidence scoring
- **Testing**: Comprehensive test suite with mocked components
- **Evaluation**: Tools for evaluating sentiment control effectiveness
- **Configuration**: Flexible YAML/JSON configuration system
- **Modern Architecture**: Clean, modular code with type hints and documentation

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Sentiment-Controlled-Text-Generation.git
   cd Sentiment-Controlled-Text-Generation
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start generating sentiment-controlled text!

#### Command Line Interface

Generate text with controlled sentiment:

```bash
python src/cli.py "The weather today is" --sentiment positive --samples 3
```

Batch processing from a file:

```bash
python src/cli.py --batch data/sample_prompts.txt --sentiment negative --output results.json
```

#### Python API

```python
from src.sentiment_generator import SentimentControlledGenerator, SentimentType

# Initialize generator
generator = SentimentControlledGenerator(model_name="gpt2")

# Generate positive sentiment text
result = generator.generate_with_sentiment(
    "The weather today is",
    SentimentType.POSITIVE
)

print(f"Generated: {result.generated_text}")
print(f"Confidence: {result.confidence:.2f}")
```

## 📁 Project Structure

```
sentiment-controlled-text-generation/
├── src/                          # Source code
│   ├── sentiment_generator.py    # Core generation logic
│   └── cli.py                    # Command-line interface
├── web_app/                      # Web application
│   └── app.py                    # Streamlit interface
├── config/                       # Configuration
│   └── config.py                 # Configuration management
├── data/                         # Data utilities
│   └── generate_data.py          # Dataset generation
├── tests/                        # Test suite
│   └── test_sentiment_generator.py
├── requirements.txt              # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Configuration

The application supports flexible configuration through YAML or JSON files:

```yaml
model:
  name: "gpt2"
  device: "auto"
  use_accelerate: true

generation:
  max_length: 100
  temperature: 0.7
  top_p: 0.9
  num_return_sequences: 1

ui:
  title: "Sentiment-Controlled Text Generation"
  max_prompt_length: 500
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python tests/test_sentiment_generator.py
```

## Evaluation

Evaluate sentiment control effectiveness:

```python
from src.sentiment_generator import SentimentControlledGenerator, SentimentType

generator = SentimentControlledGenerator()

# Test prompts and expected sentiments
test_prompts = ["The weather is", "I feel about my job"]
expected_sentiments = [SentimentType.POSITIVE, SentimentType.NEGATIVE]

# Evaluate
metrics = generator.evaluate_sentiment_control(
    test_prompts, 
    expected_sentiments
)

print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"Average Confidence: {metrics['average_confidence']:.2f}")
```

## Web Interface Features

The Streamlit web application provides:

- **Interactive Text Generation**: Easy-to-use interface for generating sentiment-controlled text
- **Real-time Analytics**: Visualizations of sentiment distribution and confidence scores
- **Parameter Tuning**: Adjustable generation parameters (temperature, top-p, etc.)
- **Generation History**: Track and analyze previous generations
- **Multiple Models**: Switch between different transformer models
- **Export Results**: Save generated text and analysis results

## Advanced Features

### Fine-tuning Support

The architecture supports fine-tuning models for specific sentiment tasks:

```python
# Example fine-tuning setup (requires additional implementation)
from transformers import Trainer, TrainingArguments

# Fine-tune on sentiment-specific data
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    eval_steps=1000,
)
```

### Zero-shot Sentiment Classification

Built-in sentiment verification using zero-shot classification:

```python
# Automatic sentiment verification
confidence = generator._verify_sentiment(
    "This is amazing!", 
    SentimentType.POSITIVE
)
```

### Batch Processing

Process multiple prompts efficiently:

```python
prompts = ["The weather is", "I feel about work", "The movie was"]
sentiments = [SentimentType.POSITIVE, SentimentType.NEGATIVE, SentimentType.NEUTRAL]

results = generator.batch_generate(prompts, sentiments)
```

## Performance Optimization

- **GPU Acceleration**: Automatic GPU detection and utilization
- **Model Caching**: Efficient model loading and caching
- **Batch Processing**: Optimized batch generation
- **Memory Management**: Efficient memory usage for large models

## 🛠️ Development

### Code Quality

The project follows modern Python best practices:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests with mocked components
- **Linting**: Black formatting and flake8 linting
- **Error Handling**: Robust error handling and logging

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest tests/`
5. Format code: `black src/ tests/`
6. Submit a pull request

## Examples

### Example 1: Positive Sentiment Generation

```python
generator = SentimentControlledGenerator()
result = generator.generate_with_sentiment(
    "The new product launch was",
    SentimentType.POSITIVE
)
# Output: "The new product launch was absolutely fantastic! The innovative features..."
```

### Example 2: Negative Sentiment Generation

```python
result = generator.generate_with_sentiment(
    "My experience with customer service was",
    SentimentType.NEGATIVE
)
# Output: "My experience with customer service was terrible. The representatives were..."
```

### Example 3: Neutral Sentiment Generation

```python
result = generator.generate_with_sentiment(
    "The weather forecast for tomorrow is",
    SentimentType.NEUTRAL
)
# Output: "The weather forecast for tomorrow is average with partly cloudy skies..."
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check internet connection and model availability
3. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Tips

- Use smaller models (distilgpt2) for faster generation
- Adjust temperature for more/less creative output
- Use batch processing for multiple generations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformers library
- OpenAI for GPT-2 model
- Streamlit for the web framework
- The open-source community for various dependencies

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the test examples
# Sentiment-Controlled-Text-Generation
