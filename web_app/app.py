"""
Streamlit web interface for sentiment-controlled text generation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_generator import SentimentControlledGenerator, SentimentType, GenerationConfig
from config.config import AppConfig, load_config


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []


def load_model():
    """Load the sentiment-controlled generator model."""
    if st.session_state.generator is None:
        with st.spinner("Loading model..."):
            try:
                st.session_state.generator = SentimentControlledGenerator(
                    model_name=st.session_state.config.model.name,
                    device=st.session_state.config.model.device,
                    use_accelerate=st.session_state.config.model.use_accelerate
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
    return True


def create_sentiment_visualization(results: List[Dict]) -> go.Figure:
    """Create visualization for sentiment analysis results."""
    if not results:
        return None
    
    # Prepare data
    sentiments = [r['sentiment'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=sentiments,
            y=confidences,
            marker_color=['green' if s == 'positive' else 'red' if s == 'negative' else 'blue' 
                         for s in sentiments],
            text=[f"{c:.2f}" for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Confidence Scores",
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Sentiment-Controlled Text Generation",
        page_icon="🎭",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🎭 Sentiment-Controlled Text Generation")
    st.markdown("Generate text with controlled sentiment using state-of-the-art AI models")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"],
            index=0
        )
        
        # Generation parameters
        st.subheader("Generation Parameters")
        max_length = st.slider("Max Length", 50, 200, st.session_state.config.generation.max_length)
        temperature = st.slider("Temperature", 0.1, 2.0, st.session_state.config.generation.temperature, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, st.session_state.config.generation.top_p, 0.1)
        num_samples = st.slider("Number of Samples", 1, 5, st.session_state.config.ui.default_num_samples)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Generation")
        
        # Input prompt
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="The weather today is...",
            height=100,
            max_chars=st.session_state.config.ui.max_prompt_length
        )
        
        # Sentiment selection
        sentiment_options = {
            "😊 Positive": SentimentType.POSITIVE,
            "😞 Negative": SentimentType.NEGATIVE,
            "😐 Neutral": SentimentType.NEUTRAL
        }
        
        selected_sentiment = st.selectbox(
            "Choose sentiment:",
            list(sentiment_options.keys()),
            index=0
        )
        
        # Generate button
        if st.button("🚀 Generate Text", type="primary"):
            if not prompt.strip():
                st.warning("Please enter a prompt!")
            else:
                if load_model():
                    with st.spinner("Generating text..."):
                        try:
                            config = GenerationConfig(
                                max_length=max_length,
                                temperature=temperature,
                                top_p=top_p,
                                num_return_sequences=num_samples
                            )
                            
                            results = []
                            for _ in range(num_samples):
                                result = st.session_state.generator.generate_with_sentiment(
                                    prompt,
                                    sentiment_options[selected_sentiment],
                                    config
                                )
                                results.append({
                                    'generated_text': result.generated_text,
                                    'sentiment': result.sentiment.value,
                                    'confidence': result.confidence,
                                    'prompt': result.prompt
                                })
                            
                            # Store results in session state
                            st.session_state.generation_history.extend(results)
                            
                            # Display results
                            st.success("Text generated successfully!")
                            
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Sample {i} (Confidence: {result['confidence']:.2f})"):
                                    st.write(result['generated_text'])
                            
                        except Exception as e:
                            st.error(f"Error generating text: {e}")
    
    with col2:
        st.header("📊 Analysis")
        
        if st.session_state.generation_history:
            # Create visualization
            fig = create_sentiment_visualization(st.session_state.generation_history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Statistics")
            total_generations = len(st.session_state.generation_history)
            avg_confidence = sum(r['confidence'] for r in st.session_state.generation_history) / total_generations
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Generations", total_generations)
            with col_b:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Sentiment distribution
            sentiment_counts = {}
            for result in st.session_state.generation_history:
                sentiment = result['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            if sentiment_counts:
                st.subheader("🎯 Sentiment Distribution")
                sentiment_df = pd.DataFrame(list(sentiment_counts.items()), 
                                          columns=['Sentiment', 'Count'])
                fig_pie = px.pie(sentiment_df, values='Count', names='Sentiment',
                               title="Generated Text Sentiment Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.generation_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Transformers, and PyTorch")


if __name__ == "__main__":
    main()
