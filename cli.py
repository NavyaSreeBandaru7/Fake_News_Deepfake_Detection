import click
import logging
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

# Import our modules
from main import FakeNewsDetector, ModelConfig, DeepfakeDetector, create_sample_dataset
from data_processor import AdvancedDataProcessor, DatasetConfig, KaggleDataLoader
from web_interface import run_streamlit_app, run_gradio_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fake_news_detector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# CLI Context
class CLIContext:
    """Context object for CLI state management"""
    
    def __init__(self):
        self.config = None
        self.detector = None
        self.data_processor = None
        self.verbose = False

pass_context = click.make_pass_decorator(CLIContext, ensure=True)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@pass_context
def cli(ctx: CLIContext, verbose: bool, config: Optional[str]):
    """
    🔍 Advanced Fake News & Deepfake Detection System
    
    A comprehensive AI-powered system for detecting misinformation and manipulated media.
    """
    ctx.verbose = verbose
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config:
        # Load configuration from file
        import yaml
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        ctx.config = config_data
        click.echo(f"✅ Loaded configuration from {config}")
    else:
        # Use default configuration
        ctx.config = {
            'model': {
                'text_model': {
                    'name': 'distilbert-base-uncased',
                    'max_length': 512,
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'num_epochs': 3
                }
            },
            'data': {
                'dataset': {
                    'min_text_length': 50,
                    'max_text_length': 5000,
                    'balance_classes': True
                }
            }
        }

@cli.command()
@click.option('--dataset', '-d', type=click.Path(exists=True), required=True,
              help='Path to training dataset (CSV format)')
@click.option('--output', '-o', type=click.Path(), default='./models/trained_model',
              help='Output directory for trained model')
@click.option('--model-name', '-m', default='distilbert-base-uncased',
              help='Pre-trained model name')
@click.option('--epochs', '-e', default=3, type=int, help='Number of training epochs')
@click.option('--batch-size', '-b', default=16, type=int, help='Training batch size')
@click.option('--learning-rate', '-lr', default=2e-5, type=float, help='Learning rate')
@click.option('--validation-split', default=0.2, type=float, help='Validation split ratio')
@pass_context
def train(ctx: CLIContext, dataset: str, output: str, model_name: str, 
          epochs: int, batch_size: int, learning_rate: float, validation_split: float):
    """
    🎯 Train a fake news detection model on your dataset
    
    Example:
        fake-news-detector train -d data/news_dataset.csv -o models/my_model
    """
    click.echo("🚀 Starting model training...")
    
    try:
        # Initialize configuration
        config = ModelConfig(
            model_name=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs
        )
        
        # Initialize detector
        detector = FakeNewsDetector(config)
        
        # Load and process data
        click.echo(f"📊 Loading dataset from {dataset}")
        texts, labels = detector.load_and_preprocess_data(dataset)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42
        )
        
        click.echo(f"📈 Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
        
        # Train model
        with click.progressbar(length=epochs, label='Training Progress') as bar:
            detector.train(train_texts, train_labels, val_texts, val_labels)
            for _ in range(epochs):
                bar.update(1)
        
        # Save model
        detector.save_model(output)
        click.echo(f"✅ Model saved to {output}")
        
        # Evaluate on validation set
        click.echo("📊 Evaluating model performance...")
        metrics = detector.evaluate(val_texts, val_labels)
        
        click.echo("\n🎯 Model Performance:")
        for metric, value in metrics.items():
            click.echo(f"  {metric.capitalize()}: {value:.4f}")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True,
              help='Path to trained model directory')
@click.option('--text', '-t', help='Text to analyze')
@click.option('--file', '-f', type=click.Path(exists=True), help='Text file to analyze')
@click.option('--batch', '-b', type=click.Path(exists=True), 
              help='CSV file with multiple texts to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', default='json', type=click.Choice(['json', 'csv', 'text']),
              help='Output format')
@pass_context
def predict(ctx: CLIContext, model: str, text: Optional[str], file: Optional[str], 
           batch: Optional[str], output: Optional[str], format: str):
    """
    🔍 Predict fake news using a trained model
    
    Examples:
        fake-news-detector predict -m models/my_model -t "Breaking news article..."
        fake-news-detector predict -m models/my_model -f article.txt -o results.json
        fake-news-detector predict -m models/my_model -b articles.csv -o results.csv
    """
    click.echo("🔍 Loading model for prediction...")
    
    try:
        # Load model
        config = ModelConfig()  # Will be loaded from saved model
        detector = FakeNewsDetector(config)
        detector.load_model(model)
        click.echo("✅ Model loaded successfully")
        
        results = []
        
        if text:
            # Single text prediction
            click.echo("📝 Analyzing text...")
            result = detector.predict(text)
            results.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['fake_probability']
            })
            
        elif file:
            # Single file prediction
            click.echo(f"📄 Analyzing file: {file}")
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = detector.predict(content)
            results.append({
                'file': file,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['fake_probability']
            })
            
        elif batch:
            # Batch prediction
            click.echo(f"📊 Processing batch file: {batch}")
            df = pd.read_csv(batch)
            
            if 'text' not in df.columns:
                click.echo("❌ Batch file must contain 'text' column", err=True)
                sys.exit(1)
            
            with click.progressbar(df['text'], label='Processing texts') as texts:
                for idx, text_content in enumerate(texts):
                    result = detector.predict(str(text_content))
                    results.append({
                        'id': idx,
                        'text_preview': str(text_content)[:100] + "...",
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'fake_probability': result['fake_probability']
                    })
        
        else:
            click.echo("❌ Must provide either --text, --file, or --batch option", err=True)
            sys.exit(1)
        
        # Output results
        if output:
            save_results(results, output, format)
            click.echo(f"💾 Results saved to {output}")
        else:
            display_results(results, format)
            
    except Exception as e:
        click.echo(f"❌ Prediction failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True,
              help='Path to trained model directory')
@click.option('--dataset', '-d', type=click.Path(exists=True), required=True,
              help='Path to test dataset')
@click.option('--output', '-o', type=click.Path(), help='Output file for evaluation report')
@pass_context
def evaluate(ctx: CLIContext, model: str, dataset: str, output: Optional[str]):
    """
    📊 Evaluate model performance on test dataset
    
    Example:
        fake-news-detector evaluate -m models/my_model -d data/test_set.csv
    """
    click.echo("📊 Starting model evaluation...")
    
    try:
        # Load model
        config = ModelConfig()
        detector = FakeNewsDetector(config)
        detector.load_model(model)
        click.echo("✅ Model loaded successfully")
        
        # Load test data
        click.echo(f"📄 Loading test dataset from {dataset}")
        test_texts, test_labels = detector.load_and_preprocess_data(dataset)
        click.echo(f"📈 Evaluating on {len(test_texts)} samples")
        
        # Evaluate
        with click.progressbar(length=len(test_texts), label='Evaluating') as bar:
            metrics = detector.evaluate(test_texts, test_labels)
            bar.update(len(test_texts))
        
        # Display results
        click.echo("\n🎯 Evaluation Results:")
        click.echo("=" * 50)
        for metric, value in metrics.items():
            click.echo(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")
        
        # Save detailed report if requested
        if output:
            report = {
                'model_path': model,
                'test_dataset': dataset,
                'test_samples': len(test_texts),
                'evaluation_timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"\n💾 Detailed report saved to {output}")
            
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--image', '-i', type=click.Path(exists=True), required=True,
              help='Path to image file')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@pass_context
def detect_deepfake(ctx: CLIContext, image: str, output: Optional[str]):
    """
    🖼️ Detect deepfakes in images
    
    Example:
        fake-news-detector detect-deepfake -i suspicious_image.jpg
    """
    click.echo("🖼️ Analyzing image for deepfake detection...")
    
    try:
        # Initialize deepfake detector
        detector = DeepfakeDetector()
        
        # Analyze image
        result = detector.detect_deepfake(image)
        
        # Display results
        click.echo(f"\n🔍 Deepfake Detection Results for: {image}")
        click.echo("=" * 50)
        click.echo(f"Prediction: {result['prediction'].upper()}")
        click.echo(f"Confidence: {result['confidence']:.2%}")
        click.echo(f"Status: {result['message']}")
        
        if result['prediction'] == 'fake':
            click.echo("🚨 POTENTIAL DEEPFAKE DETECTED!")
        elif result['prediction'] == 'real':
            click.echo("✅ IMAGE APPEARS AUTHENTIC")
        else:
            click.echo("⚠️ ANALYSIS INCONCLUSIVE")
        
        # Save results if requested
        if output:
            result_data = {
                'image_path': image,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'message': result['message'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output, 'w') as f:
                json.dump(result_data, f, indent=2)
            click.echo(f"\n💾 Results saved to {output}")
            
    except Exception as e:
        click.echo(f"❌ Deepfake detection failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--download-dir', '-d', default='./data/kaggle', 
              help='Directory to download datasets')
@click.option('--dataset', multiple=True, 
              help='Specific dataset to download (can be used multiple times)')
@pass_context
def download_data(ctx: CLIContext, download_dir: str, dataset: List[str]):
    """
    📥 Download datasets from Kaggle
    
    Example:
        fake-news-detector download-data -d ./data
    """
    click.echo("📥 Downloading datasets from Kaggle...")
    
    try:
        # Initialize Kaggle loader
        kaggle_loader = KaggleDataLoader()
        
        # Download datasets
        if dataset:
            # Download specific datasets
            for ds in dataset:
                click.echo(f"📊 Downloading {ds}...")
                # Download logic here
        else:
            # Download default datasets
            paths = kaggle_loader.download_fake_news_datasets(download_dir)
            
        click.echo(f"✅ Datasets downloaded to {download_dir}")
        
    except Exception as e:
        click.echo(f"❌ Download failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--interface', '-i', default='streamlit', 
              type=click.Choice(['streamlit', 'gradio']),
              help='Web interface type')
@click.option('--port', '-p', default=8501, type=int, help='Port number')
@click.option('--host', default='localhost', help='Host address')
@pass_context
def web(ctx: CLIContext, interface: str, port: int, host: str):
    """
    🌐 Launch web interface
    
    Example:
        fake-news-detector web -i streamlit -p 8501
    """
    click.echo(f"🌐 Starting {interface} web interface...")
    click.echo(f"🔗 Access the interface at: http://{host}:{port}")
    
    try:
        if interface == 'streamlit':
            import subprocess
            subprocess.run([
                'streamlit', 'run', 'web_interface.py',
                '--server.port', str(port),
                '--server.address', host
            ])
        elif interface == 'gradio':
            run_gradio_app()
    except KeyboardInterrupt:
        click.echo("\n👋 Web interface stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start web interface: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--size', '-s', default=1000, type=int, help='Number of samples to generate')
@click.option('--output', '-o', default='./data/sample_dataset.csv', 
              help='Output file path')
@pass_context
def generate_sample_data(ctx: CLIContext, size: int, output: str):
    """
    🎲 Generate sample dataset for testing
    
    Example:
        fake-news-detector generate-sample-data -s 1000 -o sample.csv
    """
    click.echo(f"🎲 Generating {size} sample data points...")
    
    try:
        # Generate sample dataset
        texts, labels = create_sample_dataset(size)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Save to file
        df.to_csv(output, index=False)
        click.echo(f"✅ Sample dataset saved to {output}")
        
        # Display statistics
        click.echo(f"\n📊 Dataset Statistics:")
        click.echo(f"Total samples: {len(df)}")
        click.echo(f"Real news: {len(df[df['label'] == 0])}")
        click.echo(f"Fake news: {len(df[df['label'] == 1])}")
        
    except Exception as e:
        click.echo(f"❌ Sample generation failed: {str(e)}", err=True)
        sys.exit(1)

def save_results(results: List[Dict], output_path: str, format: str):
    """Save prediction results to file"""
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    elif format == 'text':
        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Confidence: {result['confidence']:.2%}\n")
                f.write("-" * 50 + "\n")

def display_results(results: List[Dict], format: str):
    """Display prediction results to console"""
    if format == 'json':
        click.echo(json.dumps(results, indent=2))
    else:
        for i, result in enumerate(results, 1):
            click.echo(f"\n🔍 Result {i}:")
            click.echo(f"  Prediction: {result['prediction'].upper()}")
            click.echo(f"  Confidence: {result['confidence']:.2%}")
            if 'fake_probability' in result:
                click.echo(f"  Fake Probability: {result['fake_probability']:.2%}")

def main():
    """Main CLI entry point"""
    cli()

if __name__ == '__main__':
    main()
