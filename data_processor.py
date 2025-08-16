import os
import re
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
import kaggle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    text_column: str = "text"
    label_column: str = "label"
    min_text_length: int = 50
    max_text_length: int = 5000
    remove_duplicates: bool = True
    balance_classes: bool = True
    augmentation_factor: float = 1.5

class TextPreprocessor:
    """Advanced text preprocessing with linguistic feature extraction"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy English model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract advanced linguistic features"""
        if not text:
            return {}
        
        doc = self.nlp(text)
        blob = TextBlob(text)
        
        # Basic statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        features = {
            # Length features
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            
            # Punctuation features
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            
            # Sentiment features
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            
            # POS tag features
            'noun_ratio': len([token for token in doc if token.pos_ == 'NOUN']) / len(doc) if doc else 0,
            'verb_ratio': len([token for token in doc if token.pos_ == 'VERB']) / len(doc) if doc else 0,
            'adj_ratio': len([token for token in doc if token.pos_ == 'ADJ']) / len(doc) if doc else 0,
            'adv_ratio': len([token for token in doc if token.pos_ == 'ADV']) / len(doc) if doc else 0,
            
            # Named entity features
            'person_count': len([ent for ent in doc.ents if ent.label_ == 'PERSON']),
            'org_count': len([ent for ent in doc.ents if ent.label_ == 'ORG']),
            'gpe_count': len([ent for ent in doc.ents if ent.label_ == 'GPE']),
            
            # Readability features
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'stopword_ratio': len([w for w in words if w.lower() in self.stop_words]) / len(words) if words else 0,
        }
        
        return features
    
    def preprocess_text(self, text: str, extract_features: bool = False) -> Union[str, Tuple[str, Dict]]:
        """Complete text preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        
        if extract_features:
            features = self.extract_linguistic_features(cleaned_text)
            return cleaned_text, features
        
        return cleaned_text

class DataAugmentor:
    """Advanced data augmentation techniques for text data"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def synonym_replacement(self, text: str, n: int = 3) -> str:
        """Replace words with synonyms using WordNet"""
        blob = TextBlob(text)
        words = blob.words
        
        if len(words) < n:
            return text
        
        new_words = words.copy()
        random_indices = random.sample(range(len(words)), min(n, len(words)))
        
        for idx in random_indices:
            word = words[idx]
            synonyms = word.synsets
            
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                if synonym != word and '_' not in synonym:
                    new_words[idx] = synonym
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 2) -> str:
        """Randomly insert synonyms of random words"""
        words = text.split()
        
        for _ in range(n):
            if not words:
                break
                
            random_word = random.choice(words)
            blob_word = TextBlob(random_word)
            synonyms = blob_word.synsets
            
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                if '_' not in synonym:
                    random_idx = random.randint(0, len(words))
                    words.insert(random_idx, synonym)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 2) -> str:
        """Randomly swap words in the sentence"""
        words = text.split()
        
        for _ in range(n):
            if len(words) < 2:
                break
                
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment_text(self, text: str, num_aug: int = 1) -> List[str]:
        """Apply multiple augmentation techniques"""
        augmented_texts = []
        
        techniques = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.random_deletion
        ]
        
        for _ in range(num_aug):
            technique = random.choice(techniques)
            augmented_text = technique(text)
            augmented_texts.append(augmented_text)
        
        return augmented_texts

class KaggleDataLoader:
    """Load datasets from Kaggle for training"""
    
    def __init__(self):
        self.api = kaggle.KaggleApi()
        self.api.authenticate()
    
    def download_fake_news_datasets(self, data_dir: str = "./data") -> List[str]:
        """Download popular fake news datasets from Kaggle"""
        datasets = [
            "clmentbisaillon/fake-and-real-news-dataset",
            "c/fake-news",
            "hassanamin/textdb3"
        ]
        
        downloaded_paths = []
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        
        for dataset in datasets:
            try:
                logger.info(f"Downloading dataset: {dataset}")
                self.api.dataset_download_files(
                    dataset, 
                    path=str(data_path), 
                    unzip=True
                )
                downloaded_paths.append(str(data_path / dataset.split('/')[-1]))
            except Exception as e:
                logger.error(f"Failed to download {dataset}: {str(e)}")
        
        return downloaded_paths
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and standardize dataset format"""
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Standardize column names
            column_mapping = {
                'title': 'text',
                'content': 'text',
                'article': 'text',
                'news': 'text',
                'target': 'label',
                'class': 'label',
                'fake': 'label'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            if 'text' not in df.columns:
                # Combine title and text if available
                if 'title' in df.columns and 'content' in df.columns:
                    df['text'] = df['title'] + ' ' + df['content']
                else:
                    raise ValueError("No text column found")
            
            if 'label' not in df.columns:
                raise ValueError("No label column found")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
            return pd.DataFrame()

class AdvancedDataProcessor:
    """Main data processing pipeline orchestrator"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.augmentor = DataAugmentor()
        self.kaggle_loader = KaggleDataLoader()
        self.feature_scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter dataset"""
        logger.info(f"Initial dataset size: {len(df)}")
        
        # Remove null values
        df = df.dropna(subset=[self.config.text_column, self.config.label_column])
        
        # Filter by text length
        df = df[
            (df[self.config.text_column].str.len() >= self.config.min_text_length) &
            (df[self.config.text_column].str.len() <= self.config.max_text_length)
        ]
        
        # Remove duplicates
        if self.config.remove_duplicates:
            df = df.drop_duplicates(subset=[self.config.text_column])
        
        # Clean text
        df[self.config.text_column] = df[self.config.text_column].apply(
            self.preprocessor.clean_text
        )
        
        # Convert labels to binary
        unique_labels = df[self.config.label_column].unique()
        if len(unique_labels) > 2:
            # Map to binary (assuming last label is positive class)
            label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
            df[self.config.label_column] = df[self.config.label_column].map(label_map)
        
        logger.info(f"Cleaned dataset size: {len(df)}")
        return df
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset classes"""
        if not self.config.balance_classes:
            return df
        
        label_counts = df[self.config.label_column].value_counts()
        min_count = label_counts.min()
        
        balanced_dfs = []
        for label in label_counts.index:
            label_df = df[df[self.config.label_column] == label]
            balanced_dfs.append(label_df.sample(n=min_count, random_state=42))
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset size: {len(balanced_df)}")
        return balanced_df
    
    def augment_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with synthetic examples"""
        if self.config.augmentation_factor <= 1.0:
            return df
        
        logger.info("Augmenting dataset...")
        augmented_rows = []
        
        num_augmentations = int(len(df) * (self.config.augmentation_factor - 1))
        sample_indices = np.random.choice(len(df), num_augmentations, replace=True)
        
        for idx in tqdm(sample_indices, desc="Augmenting"):
            row = df.iloc[idx]
            text = row[self.config.text_column]
            label = row[self.config.label_column]
            
            augmented_texts = self.augmentor.augment_text(text, num_aug=1)
            
            for aug_text in augmented_texts:
                augmented_rows.append({
                    self.config.text_column: aug_text,
                    self.config.label_column: label
                })
        
        augmented_df = pd.DataFrame(augmented_rows)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Augmented dataset size: {len(combined_df)}")
        return combined_df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract linguistic and TF-IDF features"""
        logger.info("Extracting features...")
        
        # Extract linguistic features
        linguistic_features = []
        for text in tqdm(df[self.config.text_column], desc="Extracting linguistic features"):
            features = self.preprocessor.extract_linguistic_features(text)
            linguistic_features.append(features)
        
        linguistic_df = pd.DataFrame(linguistic_features)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(df[self.config.text_column])
        
        # Scale linguistic features
        scaled_linguistic = self.feature_scaler.fit_transform(linguistic_df.fillna(0))
        
        # Combine features
        combined_features = np.hstack([scaled_linguistic, tfidf_features.toarray()])
        
        return linguistic_df, combined_features
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Complete data processing pipeline"""
        logger.info("Starting data processing pipeline...")
        
        # Clean dataset
        df_clean = self.clean_dataset(df)
        
        # Balance classes
        df_balanced = self.balance_dataset(df_clean)
        
        # Augment dataset
        df_augmented = self.augment_dataset(df_balanced)
        
        # Extract features
        linguistic_features, combined_features = self.extract_features(df_augmented)
        
        logger.info("Data processing pipeline completed!")
        return df_augmented, combined_features, linguistic_features
    
    def create_visualization(self, df: pd.DataFrame, output_dir: str = "./visualizations"):
        """Create data visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        df[self.config.label_column].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.savefig(output_path / 'class_distribution.png')
        plt.close()
        
        # Text length distribution
        plt.figure(figsize=(12, 6))
        df['text_length'] = df[self.config.text_column].str.len()
        
        plt.subplot(1, 2, 1)
        df[df[self.config.label_column] == 0]['text_length'].hist(alpha=0.7, label='Real')
        df[df[self.config.label_column] == 1]['text_length'].hist(alpha=0.7, label='Fake')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Text Length Distribution by Class')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='text_length', by=self.config.label_column)
        plt.title('Text Length Boxplot by Class')
        
        plt.tight_layout()
        plt.savefig(output_path / 'text_length_analysis.png')
        plt.close()
        
        # Word clouds
        fake_text = ' '.join(df[df[self.config.label_column] == 1][self.config.text_column])
        real_text = ' '.join(df[df[self.config.label_column] == 0][self.config.text_column])
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        fake_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(fake_text)
        plt.imshow(fake_wordcloud, interpolation='bilinear')
        plt.title('Fake News Word Cloud')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        real_wordcloud = WordCloud(width=400, height=400, background_color='white').generate(real_text)
        plt.imshow(real_wordcloud, interpolation='bilinear')
        plt.title('Real News Word Cloud')
        plt.axis('off')
        
        plt.savefig(output_path / 'wordclouds.png')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def load_sample_datasets() -> pd.DataFrame:
    """Load sample datasets for testing"""
    # Create sample data if Kaggle datasets are not available
    sample_data = {
        'text': [
            "Breaking news: Government announces new healthcare policy",
            "Scientists discover new treatment for cancer",
            "SHOCKING: This simple trick will change your life forever!",
            "URGENT: You need to see this before it's too late!",
            "Weather forecast predicts sunny conditions for the weekend",
            "EXPOSED: The truth they don't want you to know!",
            "Local authorities report successful community event",
            "BREAKING: Secret conspiracy revealed by insider!",
            "Technology company releases quarterly earnings report",
            "MIRACLE CURE discovered by local doctor!"
        ] * 100,  # Repeat to create larger sample
        'label': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 100
    }
    
    return pd.DataFrame(sample_data)

async def main():
    """Demonstrate data processing pipeline"""
    logger.info("Starting data processing demonstration...")
    
    # Configuration
    config = DatasetConfig(
        min_text_length=20,
        max_text_length=1000,
        balance_classes=True,
        augmentation_factor=1.2
    )
    
    # Initialize processor
    processor = AdvancedDataProcessor(config)
    
    # Load sample data
    df = load_sample_datasets()
    logger.info(f"Loaded {len(df)} samples")
    
    # Process dataset
    processed_df, features, linguistic_features = processor.process_dataset(df)
    
    # Create visualizations
    processor.create_visualization(processed_df)
    
    # Save processed data
    processed_df.to_csv('./data/processed_dataset.csv', index=False)
    np.save('./data/features.npy', features)
    linguistic_features.to_csv('./data/linguistic_features.csv', index=False)
    
    logger.info("Data processing demonstration completed!")
    logger.info(f"Final dataset size: {len(processed_df)}")
    logger.info(f"Feature matrix shape: {features.shape}")

if __name__ == "__main__":
    asyncio.run(main())
