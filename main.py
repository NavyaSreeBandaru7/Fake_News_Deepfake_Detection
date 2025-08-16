import os
import sys
import logging
import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    RobertaTokenizer, RobertaModel, BertTokenizer, BertModel,
    pipeline, Trainer, TrainingArguments
)
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    model_name: str = "roberta-large"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    dropout_rate: float = 0.3
    hidden_dim: int = 768
    num_classes: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class AdvancedTextDataset(Dataset):
    """Enhanced dataset class for text classification with advanced preprocessing"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Advanced tokenization with attention masks
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiHeadAttentionClassifier(nn.Module):
    """Advanced transformer-based classifier with custom multi-head attention"""
    
    def __init__(self, config: ModelConfig):
        super(MultiHeadAttentionClassifier, self).__init__()
        self.config = config
        
        # Load pre-trained transformer
        if 'roberta' in config.model_name.lower():
            self.transformer = RobertaModel.from_pretrained(config.model_name)
        else:
            self.transformer = BertModel.from_pretrained(config.model_name)
        
        # Custom attention and classification layers
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = transformer_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply custom multi-head attention
        attended_output, attention_weights = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Layer normalization and residual connection
        attended_output = self.layer_norm(attended_output + sequence_output)
        
        # Global average pooling
        pooled_output = torch.mean(attended_output, dim=1)  # [batch, hidden]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        probabilities = self.softmax(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_weights': attention_weights
        }

class DeepfakeDetector:
    """Advanced deepfake detection using computer vision techniques"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_cnn_model()
    
    def _build_cnn_model(self) -> nn.Module:
        """Build advanced CNN model for deepfake detection"""
        model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Global average pooling and classification
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),  # Real vs Fake
            nn.Softmax(dim=1)
        )
        return model.to(self.device)
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Advanced image preprocessing with face detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Use the first detected face
                x, y, w, h = faces[0]
                face = image_rgb[y:y+h, x:x+w]
                
                # Resize to standard size
                face = cv2.resize(face, (224, 224))
                
                # Normalize
                face = face.astype(np.float32) / 255.0
                
                # Convert to tensor
                face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
                return face_tensor.to(self.device)
            
            return None
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def detect_deepfake(self, image_path: str) -> Dict[str, Union[float, str]]:
        """Detect if an image contains a deepfake"""
        preprocessed = self.preprocess_image(image_path)
        
        if preprocessed is None:
            return {"confidence": 0.0, "prediction": "error", "message": "No face detected"}
        
        with torch.no_grad():
            outputs = self.model(preprocessed)
            confidence = float(torch.max(outputs))
            prediction = "fake" if torch.argmax(outputs) == 1 else "real"
        
        return {
            "confidence": confidence,
            "prediction": prediction,
            "message": "Detection successful"
        }

class AgentCoordinator:
    """Multi-agent system coordinator for orchestrating different detection agents"""
    
    def __init__(self):
        self.text_agent = None
        self.image_agent = DeepfakeDetector()
        self.ensemble_weights = {"text": 0.7, "image": 0.3}
    
    def set_text_agent(self, agent):
        """Set the text detection agent"""
        self.text_agent = agent
    
    async def coordinate_detection(self, text_content: str = None, image_path: str = None) -> Dict:
        """Coordinate multiple agents for comprehensive fake content detection"""
        results = {"text_analysis": None, "image_analysis": None, "combined_score": 0.0}
        
        # Text analysis
        if text_content and self.text_agent:
            text_result = await self.text_agent.predict_async(text_content)
            results["text_analysis"] = text_result
        
        # Image analysis
        if image_path:
            image_result = self.image_agent.detect_deepfake(image_path)
            results["image_analysis"] = image_result
        
        # Combine results using weighted ensemble
        if results["text_analysis"] and results["image_analysis"]:
            text_score = results["text_analysis"].get("fake_probability", 0)
            image_score = 1.0 if results["image_analysis"]["prediction"] == "fake" else 0.0
            
            combined_score = (
                self.ensemble_weights["text"] * text_score +
                self.ensemble_weights["image"] * image_score
            )
            results["combined_score"] = combined_score
        
        return results

class FakeNewsDetector:
    """Main fake news detection system with advanced NLP capabilities"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.trained = False
        
        # Initialize components
        self._initialize_tokenizer()
        self._initialize_model()
        
        # Agent coordinator
        self.agent_coordinator = AgentCoordinator()
        self.agent_coordinator.set_text_agent(self)
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer based on model configuration"""
        try:
            if 'roberta' in self.config.model_name.lower():
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {str(e)}")
            # Fallback to distilbert
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def _initialize_model(self):
        """Initialize the classification model"""
        try:
            self.model = MultiHeadAttentionClassifier(self.config)
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def load_and_preprocess_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load and preprocess dataset from various formats"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            # Assume columns are 'text' and 'label' or similar
            text_columns = ['text', 'title', 'content', 'article', 'news']
            label_columns = ['label', 'target', 'class', 'fake']
            
            text_col = next((col for col in text_columns if col in df.columns), None)
            label_col = next((col for col in label_columns if col in df.columns), None)
            
            if text_col is None or label_col is None:
                raise ValueError("Could not identify text and label columns in dataset")
            
            texts = df[text_col].astype(str).tolist()
            labels = df[label_col].tolist()
            
            # Convert labels to binary if needed
            if not all(isinstance(label, int) and label in [0, 1] for label in labels):
                unique_labels = list(set(labels))
                label_map = {unique_labels[i]: i for i in range(len(unique_labels))}
                labels = [label_map[label] for label in labels]
            
            logger.info(f"Loaded {len(texts)} samples from {data_path}")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str] = None, val_labels: List[int] = None):
        """Train the fake news detection model"""
        logger.info("Starting model training...")
        
        # Create datasets
        train_dataset = AdvancedTextDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        # Setup optimizer and loss function
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        self.trained = True
        logger.info("Training completed successfully!")
    
    def predict(self, text: str) -> Dict[str, Union[float, str, int]]:
        """Predict if a text is fake news"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        with torch.no_grad():
            # Tokenize input
            encoding = self.tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=self.config.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            outputs = self.model(input_ids, attention_mask)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            return {
                'predicted_class': predicted_class,
                'prediction': 'fake' if predicted_class == 1 else 'real',
                'confidence': confidence,
                'fake_probability': float(probabilities[1]),
                'real_probability': float(probabilities[0])
            }
    
    async def predict_async(self, text: str) -> Dict[str, Union[float, str, int]]:
        """Asynchronous prediction for better performance in multi-agent systems"""
        return self.predict(text)
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = []
        true_labels = test_labels
        
        logger.info("Evaluating model performance...")
        for text in tqdm(test_texts, desc="Evaluating"):
            pred = self.predict(text)
            predictions.append(pred['predicted_class'])
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score']
        }
    
    def save_model(self, path: str):
        """Save trained model and tokenizer"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'trained': self.trained
        }, save_path / 'model.pth')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / 'tokenizer')
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and tokenizer"""
        load_path = Path(path)
        
        # Load model state
        checkpoint = torch.load(load_path / 'model.pth', map_location=self.device)
        self.config = checkpoint['config']
        self.trained = checkpoint['trained']
        
        # Reinitialize model with loaded config
        self._initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        if 'roberta' in self.config.model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(load_path / 'tokenizer')
        else:
            self.tokenizer = BertTokenizer.from_pretrained(load_path / 'tokenizer')
        
        logger.info(f"Model loaded from {path}")

def create_sample_dataset(size: int = 1000) -> Tuple[List[str], List[int]]:
    """Create a sample dataset for testing purposes"""
    import random
    
    real_news_templates = [
        "The government announced new policies regarding {topic}.",
        "Scientists have discovered {topic} through extensive research.",
        "The stock market showed {topic} trends today.",
        "Weather reports indicate {topic} conditions.",
        "Local authorities confirmed {topic} in the region."
    ]
    
    fake_news_templates = [
        "SHOCKING: {topic} will change everything you know!",
        "BREAKING: Secret {topic} revealed by insider sources!",
        "You won't believe what {topic} does to your body!",
        "URGENT: {topic} threatens national security!",
        "EXPOSED: The truth about {topic} they don't want you to know!"
    ]
    
    topics = [
        "healthcare", "economy", "technology", "climate change", "education",
        "sports", "entertainment", "politics", "science", "business"
    ]
    
    texts = []
    labels = []
    
    for _ in range(size):
        topic = random.choice(topics)
        if random.random() > 0.5:  # Real news
            template = random.choice(real_news_templates)
            text = template.format(topic=topic)
            label = 0  # Real
        else:  # Fake news
            template = random.choice(fake_news_templates)
            text = template.format(topic=topic)
            label = 1  # Fake
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels

async def main():
    """Main function demonstrating the complete system"""
    logger.info("Initializing Advanced Fake News & Deepfake Detection System...")
    
    # Configuration
    config = ModelConfig(
        model_name="distilbert-base-uncased",  # Using lighter model for demo
        max_length=256,
        batch_size=8,
        num_epochs=2
    )
    
    # Initialize detector
    detector = FakeNewsDetector(config)
    
    # Create sample dataset for demonstration
    logger.info("Creating sample dataset...")
    texts, labels = create_sample_dataset(500)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    detector.train(train_texts, train_labels)
    
    # Evaluate model
    metrics = detector.evaluate(test_texts, test_labels)
    logger.info(f"Model Performance: {metrics}")
    
    # Test predictions
    test_samples = [
        "The government announced new healthcare policies today.",
        "SHOCKING: This miracle cure will change your life forever!"
    ]
    
    for sample in test_samples:
        prediction = detector.predict(sample)
        logger.info(f"Text: {sample}")
        logger.info(f"Prediction: {prediction}")
        logger.info("-" * 50)
    
    # Test multi-agent coordination
    logger.info("Testing multi-agent coordination...")
    results = await detector.agent_coordinator.coordinate_detection(
        text_content="BREAKING: Secret government conspiracy revealed!"
    )
    logger.info(f"Multi-agent results: {results}")
    
    # Save model
    detector.save_model("./models/fake_news_detector")
    
    logger.info("System demonstration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
