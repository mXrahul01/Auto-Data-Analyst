"""
Text Models Module for Auto-Analyst Platform

This module implements comprehensive Natural Language Processing capabilities including:
- Traditional Text Vectorization (TF-IDF, Count Vectorizer, Hash Vectorizer)
- Word Embeddings (Word2Vec, GloVe, FastText, Sentence Transformers)
- Transformer Models (BERT, RoBERTa, DistilBERT, T5, GPT)
- Text Classification (Sentiment Analysis, Topic Classification, Intent Detection)
- Named Entity Recognition (NER) and Information Extraction
- Topic Modeling (LDA, NMF, BERTopic)
- Text Similarity and Semantic Search
- Language Detection and Translation
- Text Generation and Summarization
- Advanced Text Preprocessing and Feature Engineering

Features:
- Automatic text preprocessing with multilingual support
- State-of-the-art transformer model integration
- Custom text classification and NER pipelines
- Advanced topic modeling and clustering
- Real-time text processing and batch analysis
- Comprehensive evaluation metrics for NLP tasks
- Business intelligence for text analytics
- Integration with HuggingFace transformers
- Multilingual text analysis capabilities
- Text quality assessment and filtering
- Automated hyperparameter optimization
- Production-ready text processing pipelines
- A/B testing framework for text models
- Integration with MLflow for experiment tracking
"""

import asyncio
import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import re
from collections import defaultdict, Counter
import string
import math

# Core NLP libraries
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# Advanced NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

# Try to download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Advanced text processing
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
    
    # Try to load English model
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

# Transformers and modern NLP
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, pipeline,
        BertTokenizer, BertModel, BertForSequenceClassification,
        RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
        DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,
        T5Tokenizer, T5ForConditionalGeneration,
        GPT2Tokenizer, GPT2LMHeadModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Word embeddings
try:
    import gensim
    from gensim.models import Word2Vec, FastText, KeyedVectors
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Topic modeling
try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans
    TOPIC_MODELING_AVAILABLE = True
except ImportError:
    TOPIC_MODELING_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Text similarity and clustering
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from scipy.spatial.distance import pdist, squareform
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False

# Advanced text metrics
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class TextModelType(Enum):
    """Types of text models."""
    TFIDF = "tfidf"
    COUNT_VECTORIZER = "count_vectorizer"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    CUSTOM_TRANSFORMER = "custom_transformer"
    ENSEMBLE = "ensemble"

class TextTask(Enum):
    """Types of text analysis tasks."""
    CLASSIFICATION = "classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    NAMED_ENTITY_RECOGNITION = "ner"
    TEXT_SIMILARITY = "text_similarity"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    TEXT_CLUSTERING = "text_clustering"

class PreprocessingLevel(Enum):
    """Levels of text preprocessing."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class TextModelConfig:
    """Configuration for text models."""
    
    def __init__(self):
        # General settings
        self.auto_select_models = True
        self.max_models_to_try = 8
        self.enable_ensemble = True
        self.random_state = 42
        
        # Text preprocessing
        self.preprocessing_level = PreprocessingLevel.STANDARD
        self.remove_stopwords = True
        self.remove_punctuation = True
        self.lowercase = True
        self.remove_numbers = False
        self.remove_special_chars = True
        self.apply_stemming = False
        self.apply_lemmatization = True
        self.min_word_length = 2
        self.max_word_length = 50
        self.remove_urls = True
        self.remove_emails = True
        self.normalize_whitespace = True
        
        # Tokenization settings
        self.tokenizer_type = 'word'  # 'word', 'sentence', 'subword'
        self.max_sequence_length = 512
        self.truncation = True
        self.padding = True
        
        # Language settings
        self.primary_language = 'english'
        self.detect_language = True
        self.multilingual_support = True
        self.language_threshold = 0.8
        
        # Vectorization settings
        self.max_features = 10000
        self.min_df = 2
        self.max_df = 0.95
        self.ngram_range = (1, 2)
        self.use_idf = True
        self.smooth_idf = True
        self.sublinear_tf = True
        
        # Model-specific settings
        # TF-IDF
        self.tfidf_analyzer = 'word'
        self.tfidf_binary = False
        
        # Word embeddings
        self.embedding_size = 300
        self.window_size = 5
        self.min_count = 1
        self.workers = 4
        self.epochs = 10
        
        # Transformer models
        self.pretrained_model_name = 'distilbert-base-uncased'
        self.max_length = 512
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.use_gpu = torch.cuda.is_available() if 'torch' in globals() else False
        
        # Topic modeling
        self.n_topics = 'auto'  # 'auto' or integer
        self.topic_coherence_metric = 'c_v'
        self.max_topics_to_try = 20
        self.topic_words_per_topic = 10
        
        # NER settings
        self.ner_model = 'en_core_web_sm'
        self.entity_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']
        
        # Text similarity
        self.similarity_threshold = 0.7
        self.similarity_metric = 'cosine'
        
        # Classification settings
        self.cv_folds = 5
        self.test_size = 0.2
        self.stratify = True
        self.class_weight = 'balanced'
        
        # Performance settings
        self.enable_parallel = True
        self.n_jobs = -1
        self.cache_embeddings = True
        self.batch_processing = True
        
        # Quality settings
        self.min_text_length = 10
        self.max_text_length = 10000
        self.filter_duplicates = True
        self.text_quality_threshold = 0.5
        
        # Evaluation settings
        self.calculate_feature_importance = True
        self.generate_word_clouds = False
        self.enable_interpretability = True
        
        # Business settings
        self.enable_sentiment_analysis = True
        self.extract_key_phrases = True
        self.calculate_readability = True
        self.business_entity_extraction = True

@dataclass
class TextProcessingResult:
    """Result of text preprocessing."""
    processed_texts: List[str]
    original_texts: List[str]
    processing_stats: Dict[str, Any]
    language_info: Dict[str, Any]
    quality_scores: List[float]
    metadata: Dict[str, Any]

@dataclass
class TextModelResult:
    """Result of text model training and evaluation."""
    model_type: TextModelType
    task_type: TextTask
    model: Any
    vectorizer: Optional[Any]
    tokenizer: Optional[Any]
    cv_scores: List[float]
    cv_score_mean: float
    cv_score_std: float
    test_score: float
    training_time: float
    model_size: int
    feature_importance: Optional[Dict[str, float]]
    predictions: Optional[np.ndarray]
    prediction_probabilities: Optional[np.ndarray]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[Dict]
    hyperparameters: Dict[str, Any]
    vocabulary_size: int
    metadata: Dict[str, Any]

@dataclass
class TextAnalysisReport:
    """Comprehensive text analysis report."""
    report_id: str
    timestamp: datetime
    task_type: TextTask
    dataset_info: Dict[str, Any]
    preprocessing_results: TextProcessingResult
    models_evaluated: List[TextModelResult]
    best_model_result: TextModelResult
    ensemble_result: Optional[TextModelResult]
    text_statistics: Dict[str, Any]
    topic_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    entity_analysis: Dict[str, Any]
    similarity_analysis: Dict[str, Any]
    business_insights: List[str]
    recommendations: List[str]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]

class TextPreprocessor:
    """Advanced text preprocessing with multiple strategies."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords_set = set(stopwords.words(config.primary_language))
        
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        
        # Load spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not available")
    
    async def preprocess_texts(
        self,
        texts: List[str],
        custom_stopwords: Optional[List[str]] = None
    ) -> TextProcessingResult:
        """Preprocess a list of texts with comprehensive analysis."""
        try:
            logger.info(f"Preprocessing {len(texts)} texts")
            start_time = datetime.now()
            
            original_texts = texts.copy()
            
            # Filter out invalid texts
            valid_indices = []
            filtered_texts = []
            
            for i, text in enumerate(texts):
                if self._is_valid_text(text):
                    valid_indices.append(i)
                    filtered_texts.append(text)
            
            texts = filtered_texts
            logger.info(f"Filtered to {len(texts)} valid texts")
            
            # Language detection
            language_info = {}
            if self.config.detect_language and LANGDETECT_AVAILABLE:
                language_info = await self._detect_languages(texts)
            
            # Text quality assessment
            quality_scores = []
            if len(texts) > 0:
                quality_scores = await self._calculate_text_quality(texts)
            
            # Apply preprocessing steps
            processed_texts = []
            processing_stats = {
                'original_count': len(original_texts),
                'valid_count': len(texts),
                'filtered_count': len(original_texts) - len(texts),
                'avg_original_length': np.mean([len(t) for t in texts]) if texts else 0,
                'preprocessing_steps': []
            }
            
            for text in texts:
                processed_text = await self._preprocess_single_text(
                    text, custom_stopwords
                )
                processed_texts.append(processed_text)
            
            if processed_texts:
                processing_stats['avg_processed_length'] = np.mean([len(t) for t in processed_texts])
                processing_stats['length_reduction_ratio'] = 1 - (
                    processing_stats['avg_processed_length'] / processing_stats['avg_original_length']
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = TextProcessingResult(
                processed_texts=processed_texts,
                original_texts=original_texts,
                processing_stats=processing_stats,
                language_info=language_info,
                quality_scores=quality_scores,
                metadata={
                    'preprocessing_time': execution_time,
                    'config': asdict(self.config),
                    'valid_indices': valid_indices
                }
            )
            
            logger.info(f"Text preprocessing completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return TextProcessingResult(
                processed_texts=texts,
                original_texts=original_texts,
                processing_stats={'error': str(e)},
                language_info={},
                quality_scores=[],
                metadata={'error': str(e)}
            )
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets validity criteria."""
        try:
            if not isinstance(text, str):
                return False
            
            text = text.strip()
            if len(text) < self.config.min_text_length:
                return False
            
            if len(text) > self.config.max_text_length:
                return False
            
            # Check if text is mostly non-alphabetic
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.3:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _detect_languages(self, texts: List[str]) -> Dict[str, Any]:
        """Detect languages in the text collection."""
        try:
            if not texts:
                return {}
            
            # Sample texts for language detection (for performance)
            sample_size = min(100, len(texts))
            sample_texts = texts[:sample_size]
            
            detected_languages = []
            confidence_scores = []
            
            for text in sample_texts:
                try:
                    lang = detect(text)
                    langs = detect_langs(text)
                    confidence = langs[0].prob if langs else 0.0
                    
                    detected_languages.append(lang)
                    confidence_scores.append(confidence)
                except:
                    detected_languages.append('unknown')
                    confidence_scores.append(0.0)
            
            # Analyze language distribution
            lang_counts = Counter(detected_languages)
            most_common_lang = lang_counts.most_common(1)[0] if lang_counts else ('unknown', 0)
            
            return {
                'detected_languages': dict(lang_counts),
                'primary_language': most_common_lang[0],
                'language_confidence': np.mean(confidence_scores),
                'multilingual': len(lang_counts) > 1,
                'sample_size': sample_size
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return {}
    
    async def _calculate_text_quality(self, texts: List[str]) -> List[float]:
        """Calculate text quality scores."""
        try:
            quality_scores = []
            
            for text in texts:
                score = 1.0  # Start with perfect score
                
                # Length penalty/bonus
                length = len(text)
                if length < 50:
                    score *= 0.8  # Short texts
                elif length > 2000:
                    score *= 0.9  # Very long texts
                
                # Word diversity
                words = text.split()
                unique_words = len(set(words))
                diversity = unique_words / len(words) if words else 0
                score *= min(1.0, diversity + 0.5)
                
                # Check for repeated patterns
                if len(text) > 100:
                    repeated_chars = max(Counter(text).values()) / len(text)
                    if repeated_chars > 0.3:
                        score *= 0.5
                
                # Readability (if textstat available)
                if TEXTSTAT_AVAILABLE:
                    try:
                        flesch_score = textstat.flesch_reading_ease(text)
                        # Normalize to 0-1 range
                        readability_score = min(1.0, max(0.0, flesch_score / 100))
                        score = (score + readability_score) / 2
                    except:
                        pass
                
                quality_scores.append(max(0.0, min(1.0, score)))
            
            return quality_scores
            
        except Exception as e:
            logger.warning(f"Text quality calculation failed: {str(e)}")
            return [0.5] * len(texts)  # Default neutral quality
    
    async def _preprocess_single_text(
        self,
        text: str,
        custom_stopwords: Optional[List[str]] = None
    ) -> str:
        """Preprocess a single text with all configured steps."""
        try:
            # Remove URLs
            if self.config.remove_urls:
                text = self.url_pattern.sub(' ', text)
            
            # Remove email addresses
            if self.config.remove_emails:
                text = self.email_pattern.sub(' ', text)
            
            # Lowercase
            if self.config.lowercase:
                text = text.lower()
            
            # Remove numbers
            if self.config.remove_numbers:
                text = self.number_pattern.sub(' ', text)
            
            # Remove special characters
            if self.config.remove_special_chars:
                if self.config.remove_punctuation:
                    text = self.special_chars_pattern.sub(' ', text)
                else:
                    # Keep some punctuation
                    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
            
            # Normalize whitespace
            if self.config.normalize_whitespace:
                text = self.whitespace_pattern.sub(' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Filter by word length
            tokens = [
                token for token in tokens
                if self.config.min_word_length <= len(token) <= self.config.max_word_length
            ]
            
            # Remove stopwords
            if self.config.remove_stopwords:
                stopwords_to_use = self.stopwords_set
                if custom_stopwords:
                    stopwords_to_use = stopwords_to_use.union(set(custom_stopwords))
                
                tokens = [token for token in tokens if token not in stopwords_to_use]
            
            # Apply stemming or lemmatization
            if self.config.apply_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
            elif self.config.apply_lemmatization:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Join back to text
            processed_text = ' '.join(tokens)
            
            return processed_text.strip()
            
        except Exception as e:
            logger.warning(f"Single text preprocessing failed: {str(e)}")
            return text

class TextVectorizer:
    """Advanced text vectorization with multiple methods."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
        self.vectorizers = {}
        self.embeddings = {}
        
    def create_vectorizer(
        self,
        vectorizer_type: TextModelType,
        vocab_size: Optional[int] = None
    ) -> BaseEstimator:
        """Create text vectorizer based on type."""
        try:
            vocab_size = vocab_size or self.config.max_features
            
            if vectorizer_type == TextModelType.TFIDF:
                return TfidfVectorizer(
                    max_features=vocab_size,
                    min_df=self.config.min_df,
                    max_df=self.config.max_df,
                    ngram_range=self.config.ngram_range,
                    use_idf=self.config.use_idf,
                    smooth_idf=self.config.smooth_idf,
                    sublinear_tf=self.config.sublinear_tf,
                    analyzer=self.config.tfidf_analyzer,
                    binary=self.config.tfidf_binary
                )
            
            elif vectorizer_type == TextModelType.COUNT_VECTORIZER:
                return CountVectorizer(
                    max_features=vocab_size,
                    min_df=self.config.min_df,
                    max_df=self.config.max_df,
                    ngram_range=self.config.ngram_range,
                    analyzer=self.config.tfidf_analyzer,
                    binary=False
                )
            
            else:
                raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
                
        except Exception as e:
            logger.error(f"Vectorizer creation failed: {str(e)}")
            raise
    
    async def create_embeddings(
        self,
        texts: List[str],
        embedding_type: TextModelType
    ) -> np.ndarray:
        """Create text embeddings using various methods."""
        try:
            if embedding_type == TextModelType.WORD2VEC and GENSIM_AVAILABLE:
                return await self._create_word2vec_embeddings(texts)
            elif embedding_type == TextModelType.FASTTEXT and GENSIM_AVAILABLE:
                return await self._create_fasttext_embeddings(texts)
            elif embedding_type == TextModelType.SENTENCE_TRANSFORMER and SENTENCE_TRANSFORMERS_AVAILABLE:
                return await self._create_sentence_transformer_embeddings(texts)
            elif embedding_type in [TextModelType.BERT, TextModelType.ROBERTA, TextModelType.DISTILBERT]:
                return await self._create_transformer_embeddings(texts, embedding_type)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
                
        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            raise
    
    async def _create_word2vec_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create Word2Vec embeddings."""
        try:
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts]
            
            # Train Word2Vec model
            model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.config.embedding_size,
                window=self.config.window_size,
                min_count=self.config.min_count,
                workers=self.config.workers,
                epochs=self.config.epochs,
                seed=self.config.random_state
            )
            
            # Create document embeddings by averaging word vectors
            embeddings = []
            for tokens in tokenized_texts:
                vectors = [
                    model.wv[word] for word in tokens 
                    if word in model.wv
                ]
                if vectors:
                    doc_embedding = np.mean(vectors, axis=0)
                else:
                    doc_embedding = np.zeros(self.config.embedding_size)
                embeddings.append(doc_embedding)
            
            self.embeddings['word2vec'] = model
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Word2Vec embedding creation failed: {str(e)}")
            raise
    
    async def _create_fasttext_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create FastText embeddings."""
        try:
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts]
            
            # Train FastText model
            model = FastText(
                sentences=tokenized_texts,
                vector_size=self.config.embedding_size,
                window=self.config.window_size,
                min_count=self.config.min_count,
                workers=self.config.workers,
                epochs=self.config.epochs,
                seed=self.config.random_state
            )
            
            # Create document embeddings
            embeddings = []
            for tokens in tokenized_texts:
                vectors = [
                    model.wv[word] for word in tokens 
                    if word in model.wv
                ]
                if vectors:
                    doc_embedding = np.mean(vectors, axis=0)
                else:
                    doc_embedding = np.zeros(self.config.embedding_size)
                embeddings.append(doc_embedding)
            
            self.embeddings['fasttext'] = model
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"FastText embedding creation failed: {str(e)}")
            raise
    
    async def _create_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create Sentence Transformer embeddings."""
        try:
            model_name = 'all-MiniLM-L6-v2'  # Default lightweight model
            model = SentenceTransformer(model_name)
            
            # Create embeddings
            embeddings = model.encode(texts, show_progress_bar=True)
            
            self.embeddings['sentence_transformer'] = model
            return embeddings
            
        except Exception as e:
            logger.error(f"Sentence Transformer embedding creation failed: {str(e)}")
            raise
    
    async def _create_transformer_embeddings(
        self, 
        texts: List[str], 
        model_type: TextModelType
    ) -> np.ndarray:
        """Create transformer-based embeddings."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not available")
            
            # Select model based on type
            model_name_map = {
                TextModelType.BERT: 'bert-base-uncased',
                TextModelType.ROBERTA: 'roberta-base',
                TextModelType.DISTILBERT: 'distilbert-base-uncased'
            }
            
            model_name = model_name_map.get(model_type, 'distilbert-base-uncased')
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            if self.config.use_gpu and torch.cuda.is_available():
                model = model.cuda()
            
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                if self.config.use_gpu and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embedding or mean pooling
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    if self.config.use_gpu:
                        batch_embeddings = batch_embeddings.cpu()
                    
                    embeddings.extend(batch_embeddings.numpy())
            
            self.embeddings[model_type.value] = {
                'tokenizer': tokenizer,
                'model': model
            }
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Transformer embedding creation failed: {str(e)}")
            raise

class TextClassifier:
    """Advanced text classification with multiple algorithms."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
        self.vectorizer = TextVectorizer(config)
        self.models = {}
        self.label_encoder = LabelEncoder()
        
    async def train_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model_types: Optional[List[TextModelType]] = None
    ) -> List[TextModelResult]:
        """Train text classification models."""
        try:
            logger.info(f"Training text classifier on {len(texts)} samples")
            
            if model_types is None:
                model_types = self._select_default_models()
            
            # Encode labels
            y = self.label_encoder.fit_transform(labels)
            
            # Split data
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts, y,
                test_size=self.config.test_size,
                stratify=y if self.config.stratify else None,
                random_state=self.config.random_state
            )
            
            results = []
            
            for model_type in model_types:
                try:
                    result = await self._train_single_classifier(
                        model_type, X_train_texts, y_train, X_test_texts, y_test
                    )
                    if result:
                        results.append(result)
                        logger.info(f"{model_type.value} - Test Score: {result.test_score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Model {model_type.value} training failed: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Text classification training failed: {str(e)}")
            return []
    
    def _select_default_models(self) -> List[TextModelType]:
        """Select default models based on available libraries."""
        models = [TextModelType.TFIDF, TextModelType.COUNT_VECTORIZER]
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            models.append(TextModelType.SENTENCE_TRANSFORMER)
        
        if TRANSFORMERS_AVAILABLE and self.config.use_gpu:
            models.append(TextModelType.DISTILBERT)
        
        if GENSIM_AVAILABLE:
            models.extend([TextModelType.WORD2VEC, TextModelType.FASTTEXT])
        
        return models[:self.config.max_models_to_try]
    
    async def _train_single_classifier(
        self,
        model_type: TextModelType,
        X_train_texts: List[str],
        y_train: np.ndarray,
        X_test_texts: List[str],
        y_test: np.ndarray
    ) -> Optional[TextModelResult]:
        """Train a single text classification model."""
        try:
            start_time = datetime.now()
            
            # Create features
            if model_type in [TextModelType.TFIDF, TextModelType.COUNT_VECTORIZER]:
                vectorizer = self.vectorizer.create_vectorizer(model_type)
                X_train = vectorizer.fit_transform(X_train_texts)
                X_test = vectorizer.transform(X_test_texts)
                
                # Use Logistic Regression for classification
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_state,
                    max_iter=1000
                )
                
            else:
                # For embedding-based models
                X_train = await self.vectorizer.create_embeddings(X_train_texts, model_type)
                X_test = await self.vectorizer.create_embeddings(X_test_texts, model_type)
                vectorizer = None
                
                # Use Random Forest for embeddings
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(
                    n_estimators=100,
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
                )
            
            # Train model
            classifier.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                classifier, X_train, y_train,
                cv=StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state
                ),
                scoring='accuracy',
                n_jobs=1  # Avoid nested parallelization
            )
            
            # Test evaluation
            test_score = classifier.score(X_test, y_test)
            predictions = classifier.predict(X_test)
            
            # Prediction probabilities
            prediction_probabilities = None
            if hasattr(classifier, 'predict_proba'):
                prediction_probabilities = classifier.predict_proba(X_test)
            
            # Additional metrics
            confusion_mat = confusion_matrix(y_test, predictions)
            class_report = classification_report(
                y_test, predictions,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Feature importance
            feature_importance = None
            if hasattr(classifier, 'feature_importances_') and vectorizer:
                try:
                    feature_names = vectorizer.get_feature_names_out()
                    importances = classifier.feature_importances_
                    
                    # Get top features
                    top_indices = np.argsort(importances)[-20:][::-1]
                    feature_importance = {
                        feature_names[i]: float(importances[i])
                        for i in top_indices
                    }
                except:
                    pass
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate model size
            model_size = len(pickle.dumps(classifier))
            if vectorizer:
                model_size += len(pickle.dumps(vectorizer))
            
            # Vocabulary size
            vocab_size = len(vectorizer.vocabulary_) if vectorizer and hasattr(vectorizer, 'vocabulary_') else X_train.shape[1]
            
            return TextModelResult(
                model_type=model_type,
                task_type=TextTask.CLASSIFICATION,
                model=classifier,
                vectorizer=vectorizer,
                tokenizer=None,
                cv_scores=cv_scores.tolist(),
                cv_score_mean=float(np.mean(cv_scores)),
                cv_score_std=float(np.std(cv_scores)),
                test_score=float(test_score),
                training_time=training_time,
                model_size=model_size,
                feature_importance=feature_importance,
                predictions=predictions,
                prediction_probabilities=prediction_probabilities,
                confusion_matrix=confusion_mat,
                classification_report=class_report,
                hyperparameters={},
                vocabulary_size=vocab_size,
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Single classifier training failed for {model_type.value}: {str(e)}")
            return None

class TopicModeler:
    """Advanced topic modeling with multiple algorithms."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
        self.vectorizer = TextVectorizer(config)
        
    async def perform_topic_modeling(
        self,
        texts: List[str],
        n_topics: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform topic modeling on texts."""
        try:
            logger.info(f"Performing topic modeling on {len(texts)} texts")
            
            if n_topics is None:
                n_topics = await self._determine_optimal_topics(texts)
            
            results = {}
            
            # LDA topic modeling
            if TOPIC_MODELING_AVAILABLE:
                lda_result = await self._lda_topic_modeling(texts, n_topics)
                results['lda'] = lda_result
            
            # NMF topic modeling
            if TOPIC_MODELING_AVAILABLE:
                nmf_result = await self._nmf_topic_modeling(texts, n_topics)
                results['nmf'] = nmf_result
            
            # BERTopic modeling
            if BERTOPIC_AVAILABLE:
                bertopic_result = await self._bertopic_modeling(texts)
                results['bertopic'] = bertopic_result
            
            # Analyze results
            analysis = await self._analyze_topic_results(results, texts)
            results['analysis'] = analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
            return {'error': str(e)}
    
    async def _determine_optimal_topics(self, texts: List[str]) -> int:
        """Determine optimal number of topics using coherence score."""
        try:
            if self.config.n_topics != 'auto':
                return int(self.config.n_topics)
            
            # Create TF-IDF matrix
            vectorizer = self.vectorizer.create_vectorizer(TextModelType.TFIDF)
            X = vectorizer.fit_transform(texts)
            
            # Try different numbers of topics
            max_topics = min(self.config.max_topics_to_try, len(texts) // 10)
            topic_range = range(2, max_topics + 1, 2)
            
            best_score = -float('inf')
            best_n_topics = 5
            
            for n_topics in topic_range:
                try:
                    # Quick LDA model
                    lda = LatentDirichletAllocation(
                        n_components=n_topics,
                        random_state=self.config.random_state,
                        max_iter=10  # Quick iteration for evaluation
                    )
                    lda.fit(X)
                    
                    # Calculate perplexity as proxy for coherence
                    score = -lda.perplexity(X)
                    
                    if score > best_score:
                        best_score = score
                        best_n_topics = n_topics
                        
                except Exception:
                    continue
            
            logger.info(f"Optimal number of topics determined: {best_n_topics}")
            return best_n_topics
            
        except Exception as e:
            logger.warning(f"Optimal topic determination failed: {str(e)}")
            return 5  # Default
    
    async def _lda_topic_modeling(
        self, 
        texts: List[str], 
        n_topics: int
    ) -> Dict[str, Any]:
        """Perform LDA topic modeling."""
        try:
            # Create TF-IDF matrix
            vectorizer = self.vectorizer.create_vectorizer(TextModelType.TFIDF)
            X = vectorizer.fit_transform(texts)
            
            # LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.config.random_state,
                max_iter=100,
                learning_method='batch'
            )
            
            # Fit model
            doc_topic_dist = lda.fit_transform(X)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-self.config.topic_words_per_topic:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'word_weight_pairs': list(zip(top_words, top_weights))
                })
            
            # Document-topic assignments
            doc_topics = []
            for doc_idx, doc_dist in enumerate(doc_topic_dist):
                dominant_topic = np.argmax(doc_dist)
                confidence = doc_dist[dominant_topic]
                
                doc_topics.append({
                    'document_idx': doc_idx,
                    'dominant_topic': int(dominant_topic),
                    'confidence': float(confidence),
                    'topic_distribution': doc_dist.tolist()
                })
            
            return {
                'model': lda,
                'vectorizer': vectorizer,
                'topics': topics,
                'document_topics': doc_topics,
                'n_topics': n_topics,
                'perplexity': lda.perplexity(X),
                'log_likelihood': lda.score(X)
            }
            
        except Exception as e:
            logger.error(f"LDA topic modeling failed: {str(e)}")
            return {'error': str(e)}
    
    async def _nmf_topic_modeling(
        self, 
        texts: List[str], 
        n_topics: int
    ) -> Dict[str, Any]:
        """Perform NMF topic modeling."""
        try:
            # Create TF-IDF matrix (NMF works well with TF-IDF)
            vectorizer = self.vectorizer.create_vectorizer(TextModelType.TFIDF)
            X = vectorizer.fit_transform(texts)
            
            # NMF model
            nmf = NMF(
                n_components=n_topics,
                random_state=self.config.random_state,
                max_iter=100,
                alpha_W=0.1,
                alpha_H=0.1,
                l1_ratio=0.5
            )
            
            # Fit model
            doc_topic_dist = nmf.fit_transform(X)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(nmf.components_):
                top_words_idx = topic.argsort()[-self.config.topic_words_per_topic:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'word_weight_pairs': list(zip(top_words, top_weights))
                })
            
            # Document-topic assignments
            doc_topics = []
            for doc_idx, doc_dist in enumerate(doc_topic_dist):
                dominant_topic = np.argmax(doc_dist)
                confidence = doc_dist[dominant_topic] / np.sum(doc_dist) if np.sum(doc_dist) > 0 else 0
                
                doc_topics.append({
                    'document_idx': doc_idx,
                    'dominant_topic': int(dominant_topic),
                    'confidence': float(confidence),
                    'topic_distribution': doc_dist.tolist()
                })
            
            return {
                'model': nmf,
                'vectorizer': vectorizer,
                'topics': topics,
                'document_topics': doc_topics,
                'n_topics': n_topics,
                'reconstruction_error': nmf.reconstruction_err_
            }
            
        except Exception as e:
            logger.error(f"NMF topic modeling failed: {str(e)}")
            return {'error': str(e)}
    
    async def _bertopic_modeling(self, texts: List[str]) -> Dict[str, Any]:
        """Perform BERTopic modeling."""
        try:
            # Create BERTopic model
            topic_model = BERTopic(
                language="english",
                calculate_probabilities=True,
                verbose=False
            )
            
            # Fit model and get topics and probabilities
            topics, probs = topic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            
            # Extract topics with words
            topics_list = []
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # Exclude noise topic
                    topic_words = topic_model.get_topic(topic_id)
                    if topic_words:
                        words = [word for word, _ in topic_words]
                        weights = [weight for _, weight in topic_words]
                        
                        topics_list.append({
                            'topic_id': int(topic_id),
                            'words': words,
                            'weights': weights,
                            'word_weight_pairs': topic_words
                        })
            
            # Document-topic assignments
            doc_topics = []
            for doc_idx, (topic_id, prob) in enumerate(zip(topics, probs)):
                doc_topics.append({
                    'document_idx': doc_idx,
                    'dominant_topic': int(topic_id),
                    'confidence': float(prob) if prob is not None else 0.0,
                    'topic_distribution': [prob] if prob is not None else []
                })
            
            return {
                'model': topic_model,
                'topics': topics_list,
                'document_topics': doc_topics,
                'n_topics': len(topics_list),
                'topic_info': topic_info.to_dict() if hasattr(topic_info, 'to_dict') else {}
            }
            
        except Exception as e:
            logger.error(f"BERTopic modeling failed: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_topic_results(
        self,
        results: Dict[str, Any],
        texts: List[str]
    ) -> Dict[str, Any]:
        """Analyze and compare topic modeling results."""
        try:
            analysis = {
                'models_compared': list(results.keys()),
                'topic_counts': {},
                'topic_quality': {},
                'best_model': None
            }
            
            # Compare topic counts
            for model_name, result in results.items():
                if 'n_topics' in result:
                    analysis['topic_counts'][model_name] = result['n_topics']
            
            # Topic coherence/quality assessment (simplified)
            quality_scores = {}
            for model_name, result in results.items():
                if 'topics' in result and result['topics']:
                    # Simple coherence proxy: average word weight in top topics
                    avg_weight = np.mean([
                        np.mean(topic['weights'][:5])  # Top 5 words
                        for topic in result['topics'][:5]  # Top 5 topics
                        if topic['weights']
                    ])
                    quality_scores[model_name] = avg_weight
            
            analysis['topic_quality'] = quality_scores
            
            # Select best model
            if quality_scores:
                best_model = max(quality_scores, key=quality_scores.get)
                analysis['best_model'] = best_model
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Topic analysis failed: {str(e)}")
            return {}

class TextAnalysisEngine:
    """
    Comprehensive text analysis engine with multiple NLP capabilities.
    """
    
    def __init__(self, config: Optional[TextModelConfig] = None):
        self.config = config or TextModelConfig()
        self.preprocessor = TextPreprocessor(self.config)
        self.classifier = TextClassifier(self.config)
        self.topic_modeler = TopicModeler(self.config)
        self.models = {}
        self.preprocessing_result = None
        
        logger.info("TextAnalysisEngine initialized")
    
    async def analyze_texts(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        task_type: Optional[TextTask] = None,
        custom_stopwords: Optional[List[str]] = None
    ) -> TextAnalysisReport:
        """
        Comprehensive text analysis with multiple NLP tasks.
        
        Args:
            texts: List of text documents
            labels: Optional labels for supervised tasks
            task_type: Specific task to focus on
            custom_stopwords: Additional stopwords to remove
            
        Returns:
            Comprehensive text analysis report
        """
        try:
            logger.info(f"Starting text analysis on {len(texts)} documents")
            start_time = datetime.now()
            
            # Determine task type
            if task_type is None:
                if labels is not None:
                    task_type = TextTask.CLASSIFICATION
                else:
                    task_type = TextTask.TOPIC_MODELING
            
            # Text preprocessing
            preprocessing_result = await self.preprocessor.preprocess_texts(
                texts, custom_stopwords
            )
            self.preprocessing_result = preprocessing_result
            
            processed_texts = preprocessing_result.processed_texts
            if not processed_texts:
                raise ValueError("No valid texts after preprocessing")
            
            # Dataset information
            dataset_info = await self._analyze_dataset_characteristics(
                texts, processed_texts, labels
            )
            
            # Text statistics
            text_statistics = await self._calculate_text_statistics(
                texts, processed_texts
            )
            
            # Initialize results
            models_evaluated = []
            best_model_result = None
            ensemble_result = None
            topic_analysis = {}
            sentiment_analysis = {}
            entity_analysis = {}
            similarity_analysis = {}
            
            # Supervised learning tasks
            if labels is not None and task_type == TextTask.CLASSIFICATION:
                logger.info("Performing text classification")
                classification_results = await self.classifier.train_classifier(
                    processed_texts, labels
                )
                models_evaluated = classification_results
                
                if classification_results:
                    best_model_result = max(
                        classification_results, 
                        key=lambda x: x.test_score
                    )
            
            # Topic modeling
            if task_type in [TextTask.TOPIC_MODELING, TextTask.CLASSIFICATION]:
                logger.info("Performing topic modeling")
                topic_analysis = await self.topic_modeler.perform_topic_modeling(
                    processed_texts
                )
            
            # Sentiment analysis
            if self.config.enable_sentiment_analysis:
                logger.info("Performing sentiment analysis")
                sentiment_analysis = await self._perform_sentiment_analysis(texts)
            
            # Named entity recognition
            if self.config.business_entity_extraction:
                logger.info("Performing named entity recognition")
                entity_analysis = await self._perform_entity_extraction(texts)
            
            # Text similarity analysis
            if len(texts) > 1:
                logger.info("Performing similarity analysis")
                similarity_analysis = await self._perform_similarity_analysis(
                    processed_texts
                )
            
            # Generate business insights and recommendations
            insights = await self._generate_insights(
                dataset_info, text_statistics, topic_analysis, 
                sentiment_analysis, best_model_result
            )
            
            recommendations = await self._generate_recommendations(
                dataset_info, insights, best_model_result
            )
            
            # Create visualizations data
            visualizations = await self._prepare_visualizations(
                text_statistics, topic_analysis, sentiment_analysis
            )
            
            # Create comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            
            report = TextAnalysisReport(
                report_id=str(uuid.uuid4()),
                timestamp=start_time,
                task_type=task_type,
                dataset_info=dataset_info,
                preprocessing_results=preprocessing_result,
                models_evaluated=models_evaluated,
                best_model_result=best_model_result,
                ensemble_result=ensemble_result,
                text_statistics=text_statistics,
                topic_analysis=topic_analysis,
                sentiment_analysis=sentiment_analysis,
                entity_analysis=entity_analysis,
                similarity_analysis=similarity_analysis,
                business_insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
                metadata={
                    'execution_time': execution_time,
                    'config': asdict(self.config),
                    'libraries_available': {
                        'spacy': SPACY_AVAILABLE,
                        'transformers': TRANSFORMERS_AVAILABLE,
                        'gensim': GENSIM_AVAILABLE,
                        'bertopic': BERTOPIC_AVAILABLE
                    }
                }
            )
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                await self._log_to_mlflow(report)
            
            logger.info(f"Text analysis completed in {execution_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            # Return minimal report with error
            return TextAnalysisReport(
                report_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                task_type=task_type or TextTask.CLASSIFICATION,
                dataset_info={},
                preprocessing_results=TextProcessingResult(
                    processed_texts=texts,
                    original_texts=texts,
                    processing_stats={'error': str(e)},
                    language_info={},
                    quality_scores=[],
                    metadata={}
                ),
                models_evaluated=[],
                best_model_result=None,
                ensemble_result=None,
                text_statistics={},
                topic_analysis={},
                sentiment_analysis={},
                entity_analysis={},
                similarity_analysis={},
                business_insights=[f"Analysis failed: {str(e)}"],
                recommendations=["Review text data quality and configuration"],
                visualizations={},
                metadata={'error': str(e)}
            )
    
    async def _analyze_dataset_characteristics(
        self,
        original_texts: List[str],
        processed_texts: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        try:
            characteristics = {
                'n_documents': len(original_texts),
                'n_processed_documents': len(processed_texts),
                'avg_document_length': np.mean([len(text) for text in original_texts]),
                'avg_processed_length': np.mean([len(text) for text in processed_texts]),
                'total_characters': sum(len(text) for text in original_texts),
                'total_words': sum(len(text.split()) for text in processed_texts)
            }
            
            # Length distribution
            lengths = [len(text) for text in original_texts]
            characteristics['length_stats'] = {
                'min': int(min(lengths)),
                'max': int(max(lengths)),
                'median': int(np.median(lengths)),
                'std': float(np.std(lengths))
            }
            
            # Label analysis if available
            if labels:
                label_counts = Counter(labels)
                characteristics['label_distribution'] = dict(label_counts)
                characteristics['n_classes'] = len(label_counts)
                characteristics['most_common_class'] = label_counts.most_common(1)[0]
                
                # Class balance
                total_labels = len(labels)
                class_ratios = {
                    label: count / total_labels 
                    for label, count in label_counts.items()
                }
                characteristics['class_balance'] = class_ratios
                
                # Check for imbalance
                min_ratio = min(class_ratios.values())
                max_ratio = max(class_ratios.values())
                characteristics['imbalance_ratio'] = max_ratio / min_ratio
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"Dataset characteristics analysis failed: {str(e)}")
            return {}
    
    async def _calculate_text_statistics(
        self,
        original_texts: List[str],
        processed_texts: List[str]
    ) -> Dict[str, Any]:
        """Calculate comprehensive text statistics."""
        try:
            statistics = {}
            
            # Basic statistics
            all_words = ' '.join(processed_texts).split()
            word_counts = Counter(all_words)
            
            statistics['vocabulary'] = {
                'unique_words': len(word_counts),
                'total_words': len(all_words),
                'avg_word_frequency': len(all_words) / len(word_counts),
                'most_common_words': word_counts.most_common(20)
            }
            
            # Document statistics
            doc_word_counts = [len(text.split()) for text in processed_texts]
            statistics['documents'] = {
                'avg_words_per_doc': float(np.mean(doc_word_counts)),
                'std_words_per_doc': float(np.std(doc_word_counts)),
                'min_words_per_doc': int(min(doc_word_counts)),
                'max_words_per_doc': int(max(doc_word_counts))
            }
            
            # Sentence statistics (if texts are long enough)
            sentence_counts = []
            for text in original_texts:
                sentences = sent_tokenize(text)
                sentence_counts.append(len(sentences))
            
            if sentence_counts:
                statistics['sentences'] = {
                    'avg_sentences_per_doc': float(np.mean(sentence_counts)),
                    'std_sentences_per_doc': float(np.std(sentence_counts))
                }
            
            # Readability statistics (if available)
            if TEXTSTAT_AVAILABLE:
                readability_scores = []
                for text in original_texts:
                    try:
                        score = textstat.flesch_reading_ease(text)
                        readability_scores.append(score)
                    except:
                        continue
                
                if readability_scores:
                    statistics['readability'] = {
                        'avg_flesch_score': float(np.mean(readability_scores)),
                        'readability_level': textstat.flesch_reading_ease(
                            ' '.join(original_texts[:100])  # Sample
                        )
                    }
            
            return statistics
            
        except Exception as e:
            logger.warning(f"Text statistics calculation failed: {str(e)}")
            return {}
    
    async def _perform_sentiment_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Perform sentiment analysis on texts."""
        try:
            analyzer = SentimentIntensityAnalyzer()
            
            sentiments = []
            for text in texts:
                scores = analyzer.polarity_scores(text)
                sentiments.append(scores)
            
            # Aggregate sentiment statistics
            compound_scores = [s['compound'] for s in sentiments]
            
            # Classify sentiments
            sentiment_labels = []
            for score in compound_scores:
                if score >= 0.05:
                    sentiment_labels.append('positive')
                elif score <= -0.05:
                    sentiment_labels.append('negative')
                else:
                    sentiment_labels.append('neutral')
            
            sentiment_distribution = Counter(sentiment_labels)
            
            return {
                'individual_sentiments': sentiments,
                'sentiment_labels': sentiment_labels,
                'distribution': dict(sentiment_distribution),
                'statistics': {
                    'avg_compound_score': float(np.mean(compound_scores)),
                    'std_compound_score': float(np.std(compound_scores)),
                    'most_positive_idx': int(np.argmax(compound_scores)),
                    'most_negative_idx': int(np.argmin(compound_scores))
                }
            }
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {}
    
    async def _perform_entity_extraction(self, texts: List[str]) -> Dict[str, Any]:
        """Perform named entity recognition."""
        try:
            all_entities = []
            entity_counts = defaultdict(int)
            
            if SPACY_AVAILABLE and self.nlp:
                # Use spaCy for NER
                for text in texts:
                    doc = self.nlp(text)
                    doc_entities = []
                    
                    for ent in doc.ents:
                        if ent.label_ in self.config.entity_types:
                            entity_info = {
                                'text': ent.text,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char
                            }
                            doc_entities.append(entity_info)
                            entity_counts[ent.label_] += 1
                    
                    all_entities.append(doc_entities)
            
            else:
                # Fallback to NLTK NER (basic)
                for text in texts:
                    tokens = word_tokenize(text)
                    pos_tags = pos_tag(tokens)
                    tree = ne_chunk(pos_tags, binary=False)
                    
                    doc_entities = []
                    for subtree in tree:
                        if hasattr(subtree, 'label'):
                            entity_text = ' '.join([token for token, pos in subtree.leaves()])
                            entity_label = subtree.label()
                            
                            if entity_label in self.config.entity_types:
                                entity_info = {
                                    'text': entity_text,
                                    'label': entity_label,
                                    'start': -1,
                                    'end': -1
                                }
                                doc_entities.append(entity_info)
                                entity_counts[entity_label] += 1
                    
                    all_entities.append(doc_entities)
            
            # Get most common entities
            all_entity_texts = []
            for doc_entities in all_entities:
                for entity in doc_entities:
                    all_entity_texts.append(entity['text'])
            
            entity_frequency = Counter(all_entity_texts)
            
            return {
                'entities_per_document': all_entities,
                'entity_type_counts': dict(entity_counts),
                'most_common_entities': entity_frequency.most_common(20),
                'total_entities': sum(entity_counts.values()),
                'unique_entities': len(entity_frequency)
            }
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}")
            return {}
    
    async def _perform_similarity_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Perform text similarity analysis."""
        try:
            if not SIMILARITY_AVAILABLE or len(texts) < 2:
                return {}
            
            # Create TF-IDF vectors for similarity
            vectorizer = TfidfVectorizer(
                max_features=min(1000, self.config.max_features),
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find most similar document pairs
            n_docs = len(texts)
            similar_pairs = []
            
            for i in range(n_docs):
                for j in range(i + 1, n_docs):
                    similarity = similarity_matrix[i, j]
                    if similarity > self.config.similarity_threshold:
                        similar_pairs.append({
                            'doc1_idx': i,
                            'doc2_idx': j,
                            'similarity': float(similarity)
                        })
            
            # Sort by similarity
            similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Overall similarity statistics
            upper_triangle = similarity_matrix[np.triu_indices(n_docs, k=1)]
            
            return {
                'similarity_matrix': similarity_matrix.tolist(),
                'similar_pairs': similar_pairs[:20],  # Top 20 similar pairs
                'statistics': {
                    'avg_similarity': float(np.mean(upper_triangle)),
                    'max_similarity': float(np.max(upper_triangle)),
                    'min_similarity': float(np.min(upper_triangle)),
                    'std_similarity': float(np.std(upper_triangle))
                },
                'high_similarity_count': len(similar_pairs)
            }
            
        except Exception as e:
            logger.warning(f"Similarity analysis failed: {str(e)}")
            return {}
    
    async def _generate_insights(
        self,
        dataset_info: Dict[str, Any],
        text_statistics: Dict[str, Any],
        topic_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        best_model_result: Optional[TextModelResult]
    ) -> List[str]:
        """Generate business insights from text analysis."""
        try:
            insights = []
            
            # Dataset insights
            n_docs = dataset_info.get('n_documents', 0)
            if n_docs > 0:
                avg_length = dataset_info.get('avg_document_length', 0)
                
                if avg_length < 100:
                    insights.append(f"Short documents detected (avg: {avg_length:.0f} chars) - consider aggregation or context enhancement")
                elif avg_length > 5000:
                    insights.append(f"Long documents detected (avg: {avg_length:.0f} chars) - consider chunking or summarization")
                
                # Class imbalance insights
                if 'imbalance_ratio' in dataset_info:
                    imbalance = dataset_info['imbalance_ratio']
                    if imbalance > 3:
                        insights.append(f"Significant class imbalance detected (ratio: {imbalance:.1f}) - consider balancing techniques")
            
            # Vocabulary insights
            if 'vocabulary' in text_statistics:
                vocab_size = text_statistics['vocabulary']['unique_words']
                total_words = text_statistics['vocabulary']['total_words']
                
                vocab_ratio = vocab_size / total_words if total_words > 0 else 0
                
                if vocab_ratio > 0.5:
                    insights.append("High vocabulary diversity detected - rich content but may need more data for stable models")
                elif vocab_ratio < 0.1:
                    insights.append("Low vocabulary diversity - content may be repetitive or domain-specific")
            
            # Model performance insights
            if best_model_result:
                score = best_model_result.test_score
                model_type = best_model_result.model_type.value
                
                if score > 0.9:
                    insights.append(f"Excellent model performance ({score:.3f}) achieved with {model_type}")
                elif score > 0.8:
                    insights.append(f"Good model performance ({score:.3f}) with {model_type} - minor improvements possible")
                elif score < 0.7:
                    insights.append(f"Moderate performance ({score:.3f}) - consider feature engineering or more advanced models")
            
            # Topic modeling insights
            if 'analysis' in topic_analysis and 'best_model' in topic_analysis['analysis']:
                best_topic_model = topic_analysis['analysis']['best_model']
                n_topics = topic_analysis['analysis']['topic_counts'].get(best_topic_model, 0)
                
                if n_topics > 0:
                    insights.append(f"Identified {n_topics} main topics using {best_topic_model} - content shows clear thematic structure")
            
            # Sentiment insights
            if sentiment_analysis and 'distribution' in sentiment_analysis:
                sentiment_dist = sentiment_analysis['distribution']
                total_docs = sum(sentiment_dist.values())
                
                if total_docs > 0:
                    positive_ratio = sentiment_dist.get('positive', 0) / total_docs
                    negative_ratio = sentiment_dist.get('negative', 0) / total_docs
                    
                    if positive_ratio > 0.6:
                        insights.append(f"Predominantly positive sentiment ({positive_ratio:.1%}) in the text collection")
                    elif negative_ratio > 0.6:
                        insights.append(f"Predominantly negative sentiment ({negative_ratio:.1%}) - may indicate issues or concerns")
                    else:
                        insights.append("Balanced sentiment distribution - neutral or mixed opinions")
            
            # Default insight
            if not insights:
                insights.append("Text analysis completed successfully - review detailed results for patterns")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {str(e)}")
            return ["Text analysis completed - review results manually"]
    
    async def _generate_recommendations(
        self,
        dataset_info: Dict[str, Any],
        insights: List[str],
        best_model_result: Optional[TextModelResult]
    ) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Data quality recommendations
            n_docs = dataset_info.get('n_documents', 0)
            
            if n_docs < 100:
                recommendations.append("Small dataset detected - consider collecting more data for robust models")
            elif n_docs < 1000:
                recommendations.append("Medium dataset - suitable for traditional ML but consider more data for deep learning")
            
            # Class imbalance recommendations
            if 'imbalance_ratio' in dataset_info:
                imbalance = dataset_info['imbalance_ratio']
                if imbalance > 5:
                    recommendations.append("Address class imbalance using sampling techniques, cost-sensitive learning, or ensemble methods")
            
            # Model performance recommendations
            if best_model_result:
                score = best_model_result.test_score
                
                if score < 0.8:
                    recommendations.append("Consider advanced preprocessing: lemmatization, custom stopwords, or domain-specific tokenization")
                    recommendations.append("Try ensemble methods or transformer-based models for improved performance")
                
                if score > 0.95:
                    recommendations.append("Excellent performance - monitor for overfitting and validate on new data")
            
            # Text preprocessing recommendations
            if any('diversity' in insight for insight in insights):
                recommendations.append("High vocabulary diversity - consider dimensionality reduction or feature selection")
            
            if any('repetitive' in insight for insight in insights):
                recommendations.append("Low diversity content - focus on domain-specific features and expert knowledge")
            
            # Business recommendations
            recommendations.append("Implement monitoring for text data drift and model performance degradation")
            recommendations.append("Consider A/B testing different text preprocessing strategies")
            
            if len(recommendations) < 3:
                recommendations.append("Text analysis shows good results - proceed with model deployment and monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {str(e)}")
            return ["Review analysis results and consider domain expertise for improvements"]
    
    async def _prepare_visualizations(
        self,
        text_statistics: Dict[str, Any],
        topic_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for visualizations."""
        try:
            visualizations = {}
            
            # Word frequency visualization
            if 'vocabulary' in text_statistics:
                most_common = text_statistics['vocabulary'].get('most_common_words', [])
                if most_common:
                    words, counts = zip(*most_common[:20])
                    visualizations['word_frequency'] = {
                        'type': 'bar_chart',
                        'data': {
                            'words': list(words),
                            'counts': list(counts),
                            'title': 'Most Frequent Words'
                        }
                    }
            
            # Topic visualization
            if 'lda' in topic_analysis and 'topics' in topic_analysis['lda']:
                topics = topic_analysis['lda']['topics']
                topic_data = []
                
                for topic in topics[:5]:  # Top 5 topics
                    words = topic.get('words', [])[:10]  # Top 10 words
                    weights = topic.get('weights', [])[:10]
                    
                    topic_data.append({
                        'topic_id': topic['topic_id'],
                        'words': words,
                        'weights': weights
                    })
                
                visualizations['topics'] = {
                    'type': 'topic_chart',
                    'data': {
                        'topics': topic_data,
                        'title': 'Topic Modeling Results'
                    }
                }
            
            # Sentiment distribution
            if sentiment_analysis and 'distribution' in sentiment_analysis:
                sentiment_dist = sentiment_analysis['distribution']
                
                visualizations['sentiment_distribution'] = {
                    'type': 'pie_chart',
                    'data': {
                        'labels': list(sentiment_dist.keys()),
                        'values': list(sentiment_dist.values()),
                        'title': 'Sentiment Distribution'
                    }
                }
            
            return visualizations
            
        except Exception as e:
            logger.warning(f"Visualization preparation failed: {str(e)}")
            return {}
    
    async def _log_to_mlflow(self, report: TextAnalysisReport):
        """Log text analysis results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"text_analysis_{report.task_type.value}"):
                # Log parameters
                mlflow.log_param("task_type", report.task_type.value)
                mlflow.log_param("n_documents", report.dataset_info.get('n_documents', 0))
                mlflow.log_param("preprocessing_level", self.config.preprocessing_level.value)
                
                # Log dataset metrics
                if report.dataset_info:
                    for key, value in report.dataset_info.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"dataset_{key}", value)
                
                # Log model metrics
                if report.best_model_result:
                    mlflow.log_metric("best_test_score", report.best_model_result.test_score)
                    mlflow.log_metric("best_cv_score", report.best_model_result.cv_score_mean)
                    mlflow.log_param("best_model_type", report.best_model_result.model_type.value)
                
                # Log text statistics
                if 'vocabulary' in report.text_statistics:
                    vocab_stats = report.text_statistics['vocabulary']
                    mlflow.log_metric("vocabulary_size", vocab_stats.get('unique_words', 0))
                    mlflow.log_metric("total_words", vocab_stats.get('total_words', 0))
                
                # Log topic modeling results
                if 'analysis' in report.topic_analysis:
                    topic_counts = report.topic_analysis['analysis'].get('topic_counts', {})
                    for model_name, n_topics in topic_counts.items():
                        mlflow.log_metric(f"topics_{model_name}", n_topics)
                
                # Log artifacts
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                
                with open("text_analysis_report.json", "w") as f:
                    json.dump(report_dict, f, indent=2, default=str)
                mlflow.log_artifact("text_analysis_report.json")
                
                logger.info("Text analysis results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    async def predict(
        self,
        texts: List[str],
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """Make predictions on new texts using trained models."""
        try:
            if not self.models:
                raise ValueError("No trained models available")
            
            # Preprocess texts if requested
            if preprocess:
                preprocessing_result = await self.preprocessor.preprocess_texts(texts)
                processed_texts = preprocessing_result.processed_texts
            else:
                processed_texts = texts
            
            predictions = {}
            
            # Use best model for predictions
            # This would require implementing model persistence and loading
            # For now, return placeholder
            predictions = {
                'texts_processed': len(processed_texts),
                'preprocessing_applied': preprocess,
                'models_available': len(self.models)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Text prediction failed: {str(e)}")
            return {'error': str(e)}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis capabilities."""
        try:
            return {
                'preprocessing_configured': self.preprocessor is not None,
                'models_trained': len(self.models),
                'available_libraries': {
                    'spacy': SPACY_AVAILABLE,
                    'transformers': TRANSFORMERS_AVAILABLE,
                    'gensim': GENSIM_AVAILABLE,
                    'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                    'bertopic': BERTOPIC_AVAILABLE,
                    'langdetect': LANGDETECT_AVAILABLE
                },
                'supported_tasks': [task.value for task in TextTask],
                'supported_models': [model.value for model in TextModelType],
                'configuration': asdict(self.config)
            }
            
        except Exception as e:
            logger.error(f"Analysis summary generation failed: {str(e)}")
            return {'error': str(e)}

# Utility functions

def create_text_analyzer(
    preprocessing_level: str = 'standard',
    enable_advanced_models: bool = True,
    use_gpu: bool = False
) -> TextAnalysisEngine:
    """Factory function to create a TextAnalysisEngine."""
    config = TextModelConfig()
    config.preprocessing_level = PreprocessingLevel(preprocessing_level)
    config.use_gpu = use_gpu
    
    # Adjust models based on capabilities
    if not enable_advanced_models:
        config.max_models_to_try = 3
    
    return TextAnalysisEngine(config)

async def quick_text_analysis(
    texts: List[str],
    labels: Optional[List[str]] = None,
    task: str = 'auto'
) -> Dict[str, Any]:
    """Quick text analysis for simple use cases."""
    # Create analyzer with simplified configuration
    analyzer = create_text_analyzer(preprocessing_level='standard', enable_advanced_models=False)
    
    # Determine task
    if task == 'auto':
        task_type = TextTask.CLASSIFICATION if labels else TextTask.TOPIC_MODELING
    else:
        task_type = TextTask(task)
    
    # Run analysis
    report = await analyzer.analyze_texts(texts, labels, task_type)
    
    # Return simplified results
    return {
        'task_type': report.task_type.value,
        'n_documents': report.dataset_info.get('n_documents', 0),
        'best_model': report.best_model_result.model_type.value if report.best_model_result else None,
        'best_score': report.best_model_result.test_score if report.best_model_result else None,
        'insights': report.business_insights[:3],
        'recommendations': report.recommendations[:3],
        'preprocessing_stats': report.preprocessing_results.processing_stats
    }

def get_available_text_models() -> Dict[str, bool]:
    """Get available text processing capabilities."""
    return {
        'traditional_vectorizers': True,
        'word_embeddings': GENSIM_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
        'transformer_models': TRANSFORMERS_AVAILABLE,
        'topic_modeling': TOPIC_MODELING_AVAILABLE,
        'advanced_topic_modeling': BERTOPIC_AVAILABLE,
        'named_entity_recognition': SPACY_AVAILABLE,
        'language_detection': LANGDETECT_AVAILABLE,
        'advanced_preprocessing': SPACY_AVAILABLE,
        'readability_metrics': TEXTSTAT_AVAILABLE,
        'hyperparameter_optimization': OPTUNA_AVAILABLE
    }

def get_text_analysis_recommendations(
    n_documents: int,
    avg_document_length: int,
    has_labels: bool,
    domain: str = 'general'
) -> Dict[str, str]:
    """Get recommendations for text analysis configuration."""
    recommendations = {}
    
    # Document count recommendations
    if n_documents < 100:
        recommendations['data_size'] = "Small dataset - focus on simple models and careful validation"
        recommendations['models'] = "TF-IDF, Count Vectorizer, simple classifiers"
    elif n_documents < 1000:
        recommendations['data_size'] = "Medium dataset - suitable for most traditional NLP methods"
        recommendations['models'] = "TF-IDF, Word2Vec, basic transformer models"
    else:
        recommendations['data_size'] = "Large dataset - can leverage advanced transformer models"
        recommendations['models'] = "BERT, RoBERTa, advanced ensemble methods"
    
    # Document length recommendations
    if avg_document_length < 100:
        recommendations['preprocessing'] = "Short texts - minimal preprocessing, preserve context"
    elif avg_document_length > 5000:
        recommendations['preprocessing'] = "Long texts - consider chunking or summarization"
    else:
        recommendations['preprocessing'] = "Standard preprocessing recommended"
    
    # Task recommendations
    if has_labels:
        recommendations['task'] = "Supervised learning - focus on classification performance"
    else:
        recommendations['task'] = "Unsupervised learning - topic modeling and clustering"
    
    # Domain-specific recommendations
    if domain == 'medical':
        recommendations['domain'] = "Medical domain - use domain-specific models and terminology"
    elif domain == 'legal':
        recommendations['domain'] = "Legal domain - preserve formal language structure"
    elif domain == 'social_media':
        recommendations['domain'] = "Social media - handle informal language and abbreviations"
    
    return recommendations

# Example usage and testing
if __name__ == "__main__":
    async def test_text_analysis():
        """Test the text analysis functionality."""
        print("Testing Text Analysis Engine...")
        print("Available capabilities:", get_available_text_models())
        
        # Create sample text data
        sample_texts = [
            "This is a great product! I absolutely love it. Highly recommended for everyone.",
            "The service was terrible. Very disappointed with the quality and customer support.",
            "Average experience. Nothing special but not bad either. Could be improved.",
            "Excellent quality and fast delivery. Will definitely order again soon.",
            "Poor value for money. The product broke after just one week of use.",
            "Good customer service and helpful staff. Product works as expected.",
            "Not worth the price. Found better alternatives elsewhere for cheaper.",
            "Amazing results! This exceeded all my expectations. Five stars!",
            "Decent product but shipping took too long. Otherwise satisfactory experience.",
            "Outstanding quality and design. Truly impressed with this purchase."
        ]
        
        # Sample labels for classification
        sample_labels = [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'negative', 'positive', 'neutral', 'positive'
        ]
        
        print(f"Sample data: {len(sample_texts)} texts with labels")
        
        # Test quick analysis
        print("\n=== Quick Text Analysis Test ===")
        quick_results = await quick_text_analysis(sample_texts, sample_labels)
        
        print(f"Quick Analysis Results:")
        print(f"  Task Type: {quick_results['task_type']}")
        print(f"  Documents: {quick_results['n_documents']}")
        print(f"  Best Model: {quick_results['best_model']}")
        print(f"  Best Score: {quick_results['best_score']:.4f}" if quick_results['best_score'] else "No score")
        
        # Test comprehensive analysis
        print("\n=== Comprehensive Text Analysis Test ===")
        analyzer = create_text_analyzer(
            preprocessing_level='standard',
            enable_advanced_models=False  # For testing
        )
        
        report = await analyzer.analyze_texts(
            sample_texts,
            sample_labels,
            TextTask.CLASSIFICATION
        )
        
        print(f"Comprehensive Analysis Results:")
        print(f"  Task Type: {report.task_type.value}")
        print(f"  Documents Processed: {len(report.preprocessing_results.processed_texts)}")
        print(f"  Models Evaluated: {len(report.models_evaluated)}")
        
        if report.best_model_result:
            print(f"  Best Model: {report.best_model_result.model_type.value}")
            print(f"  Best Score: {report.best_model_result.test_score:.4f}")
            print(f"  CV Score: {report.best_model_result.cv_score_mean:.4f} (±{report.best_model_result.cv_score_std:.4f})")
        
        # Text statistics
        if report.text_statistics:
            print(f"\n  Text Statistics:")
            if 'vocabulary' in report.text_statistics:
                vocab = report.text_statistics['vocabulary']
                print(f"    Vocabulary Size: {vocab.get('unique_words', 0)}")
                print(f"    Total Words: {vocab.get('total_words', 0)}")
            
        # Topic analysis results
        if report.topic_analysis and 'analysis' in report.topic_analysis:
            print(f"\n  Topic Analysis:")
            analysis = report.topic_analysis['analysis']
            if 'best_model' in analysis:
                print(f"    Best Topic Model: {analysis['best_model']}")
            if 'topic_counts' in analysis:
                for model, count in analysis['topic_counts'].items():
                    print(f"    {model}: {count} topics")
        
        # Sentiment analysis
        if report.sentiment_analysis and 'distribution' in report.sentiment_analysis:
            print(f"\n  Sentiment Analysis:")
            dist = report.sentiment_analysis['distribution']
            for sentiment, count in dist.items():
                print(f"    {sentiment}: {count} documents")
        
        # Business insights
        print(f"\n  Business Insights:")
        for i, insight in enumerate(report.business_insights[:3], 1):
            print(f"    {i}. {insight}")
        
        print(f"\n  Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        # Test topic modeling without labels
        print("\n=== Topic Modeling Test ===")
        topic_report = await analyzer.analyze_texts(
            sample_texts,
            task_type=TextTask.TOPIC_MODELING
        )
        
        print(f"Topic Modeling Results:")
        if topic_report.topic_analysis and 'lda' in topic_report.topic_analysis:
            lda_result = topic_report.topic_analysis['lda']
            if 'topics' in lda_result:
                print(f"  Found {len(lda_result['topics'])} topics:")
                for topic in lda_result['topics'][:3]:
                    words = topic.get('words', [])[:5]
                    print(f"    Topic {topic['topic_id']}: {', '.join(words)}")
        
        # Test analyzer capabilities
        print(f"\n=== Analyzer Capabilities ===")
        summary = analyzer.get_analysis_summary()
        print(f"  Models Trained: {summary['models_trained']}")
        
        print(f"  Available Libraries:")
        for lib, available in summary['available_libraries'].items():
            print(f"    {lib}: {'✓' if available else '✗'}")
        
        # Test text recommendations
        print(f"\n=== Text Analysis Recommendations ===")
        recommendations = get_text_analysis_recommendations(
            n_documents=len(sample_texts),
            avg_document_length=np.mean([len(text) for text in sample_texts]),
            has_labels=True,
            domain='general'
        )
        
        for category, recommendation in recommendations.items():
            print(f"  {category}: {recommendation}")
        
        return report
    
    # Run test
    import asyncio
    results = asyncio.run(test_text_analysis())

class AdvancedTextFeatureExtractor:
    """Extract advanced features from text data."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
        
    async def extract_features(self, texts: List[str]) -> Dict[str, Any]:
        """Extract comprehensive text features."""
        try:
            features = {}
            
            # Basic text features
            features['basic'] = await self._extract_basic_features(texts)
            
            # Linguistic features
            features['linguistic'] = await self._extract_linguistic_features(texts)
            
            # Readability features
            if TEXTSTAT_AVAILABLE:
                features['readability'] = await self._extract_readability_features(texts)
            
            # Stylometric features
            features['stylometric'] = await self._extract_stylometric_features(texts)
            
            # Semantic features
            features['semantic'] = await self._extract_semantic_features(texts)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_basic_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract basic text features."""
        try:
            features = {
                'char_count': [],
                'word_count': [],
                'sentence_count': [],
                'avg_word_length': [],
                'avg_sentence_length': [],
                'punctuation_ratio': [],
                'uppercase_ratio': [],
                'digit_ratio': []
            }
            
            for text in texts:
                # Character count
                features['char_count'].append(len(text))
                
                # Word count
                words = text.split()
                features['word_count'].append(len(words))
                
                # Sentence count
                sentences = sent_tokenize(text)
                features['sentence_count'].append(len(sentences))
                
                # Average word length
                if words:
                    avg_word_len = np.mean([len(word) for word in words])
                    features['avg_word_length'].append(avg_word_len)
                else:
                    features['avg_word_length'].append(0)
                
                # Average sentence length
                if sentences:
                    avg_sent_len = np.mean([len(sent.split()) for sent in sentences])
                    features['avg_sentence_length'].append(avg_sent_len)
                else:
                    features['avg_sentence_length'].append(0)
                
                # Punctuation ratio
                punct_count = sum(1 for c in text if c in string.punctuation)
                features['punctuation_ratio'].append(punct_count / len(text) if text else 0)
                
                # Uppercase ratio
                upper_count = sum(1 for c in text if c.isupper())
                features['uppercase_ratio'].append(upper_count / len(text) if text else 0)
                
                # Digit ratio
                digit_count = sum(1 for c in text if c.isdigit())
                features['digit_ratio'].append(digit_count / len(text) if text else 0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Basic feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_linguistic_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract linguistic features using NLTK."""
        try:
            features = {
                'noun_ratio': [],
                'verb_ratio': [],
                'adjective_ratio': [],
                'adverb_ratio': [],
                'lexical_diversity': [],
                'function_word_ratio': []
            }
            
            function_words = set([
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                'do', 'at', 'this', 'but', 'his', 'by', 'from'
            ])
            
            for text in texts:
                words = word_tokenize(text.lower())
                
                if not words:
                    # Fill with zeros if no words
                    for key in features:
                        features[key].append(0.0)
                    continue
                
                # POS tagging
                pos_tags = pos_tag(words)
                
                # Count POS categories
                pos_counts = Counter(tag for _, tag in pos_tags)
                total_tags = len(pos_tags)
                
                # Noun ratio (NN, NNS, NNP, NNPS)
                noun_count = sum(count for tag, count in pos_counts.items() if tag.startswith('NN'))
                features['noun_ratio'].append(noun_count / total_tags)
                
                # Verb ratio (VB, VBD, VBG, VBN, VBP, VBZ)
                verb_count = sum(count for tag, count in pos_counts.items() if tag.startswith('VB'))
                features['verb_ratio'].append(verb_count / total_tags)
                
                # Adjective ratio (JJ, JJR, JJS)
                adj_count = sum(count for tag, count in pos_counts.items() if tag.startswith('JJ'))
                features['adjective_ratio'].append(adj_count / total_tags)
                
                # Adverb ratio (RB, RBR, RBS)
                adv_count = sum(count for tag, count in pos_counts.items() if tag.startswith('RB'))
                features['adverb_ratio'].append(adv_count / total_tags)
                
                # Lexical diversity (unique words / total words)
                unique_words = len(set(words))
                features['lexical_diversity'].append(unique_words / len(words))
                
                # Function word ratio
                function_count = sum(1 for word in words if word in function_words)
                features['function_word_ratio'].append(function_count / len(words))
            
            return features
            
        except Exception as e:
            logger.warning(f"Linguistic feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_readability_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract readability features using textstat."""
        try:
            features = {
                'flesch_reading_ease': [],
                'flesch_kincaid_grade': [],
                'gunning_fog': [],
                'automated_readability_index': [],
                'coleman_liau_index': [],
                'reading_time': []
            }
            
            for text in texts:
                try:
                    features['flesch_reading_ease'].append(textstat.flesch_reading_ease(text))
                    features['flesch_kincaid_grade'].append(textstat.flesch_kincaid(text))
                    features['gunning_fog'].append(textstat.gunning_fog(text))
                    features['automated_readability_index'].append(textstat.automated_readability_index(text))
                    features['coleman_liau_index'].append(textstat.coleman_liau_index(text))
                    features['reading_time'].append(textstat.reading_time(text, ms_per_char=14.69))
                except:
                    # Fill with default values if calculation fails
                    for key in features:
                        features[key].append(0.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Readability feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_stylometric_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract stylometric features for authorship analysis."""
        try:
            features = {
                'avg_words_per_sentence': [],
                'sentence_length_variation': [],
                'word_length_variation': [],
                'hapax_legomena_ratio': [],  # Words that appear only once
                'yules_k': [],  # Yule's K measure of lexical diversity
                'simpsons_index': []  # Simpson's diversity index
            }
            
            for text in texts:
                words = word_tokenize(text.lower())
                sentences = sent_tokenize(text)
                
                if not words or not sentences:
                    for key in features:
                        features[key].append(0.0)
                    continue
                
                # Average words per sentence
                features['avg_words_per_sentence'].append(len(words) / len(sentences))
                
                # Sentence length variation
                sent_lengths = [len(sent.split()) for sent in sentences]
                features['sentence_length_variation'].append(np.std(sent_lengths))
                
                # Word length variation
                word_lengths = [len(word) for word in words]
                features['word_length_variation'].append(np.std(word_lengths))
                
                # Hapax legomena ratio
                word_freq = Counter(words)
                hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
                features['hapax_legomena_ratio'].append(hapax_count / len(words))
                
                # Yule's K
                try:
                    n = len(words)
                    v1 = len(word_freq)  # Number of unique words
                    v2 = sum(freq * freq for freq in word_freq.values())
                    yules_k = 10000 * (v2 - v1) / (n * n) if n > 0 else 0
                    features['yules_k'].append(yules_k)
                except:
                    features['yules_k'].append(0.0)
                
                # Simpson's diversity index
                try:
                    n = len(words)
                    simpson = sum((freq * (freq - 1)) for freq in word_freq.values()) / (n * (n - 1)) if n > 1 else 0
                    features['simpsons_index'].append(1 - simpson)  # Simpson's diversity
                except:
                    features['simpsons_index'].append(0.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Stylometric feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_semantic_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """Extract semantic features using word embeddings if available."""
        try:
            features = {
                'semantic_coherence': [],
                'semantic_complexity': [],
                'topic_consistency': []
            }
            
            # This is a simplified implementation
            # In practice, would use actual embeddings and semantic models
            
            for text in texts:
                words = word_tokenize(text.lower())
                
                if not words:
                    for key in features:
                        features[key].append(0.0)
                    continue
                
                # Simplified semantic coherence (based on word repetition)
                word_freq = Counter(words)
                max_freq = max(word_freq.values()) if word_freq else 0
                coherence = max_freq / len(words) if words else 0
                features['semantic_coherence'].append(coherence)
                
                # Simplified semantic complexity (based on vocabulary diversity)
                unique_words = len(set(words))
                complexity = unique_words / len(words) if words else 0
                features['semantic_complexity'].append(complexity)
                
                # Simplified topic consistency (based on word co-occurrence)
                # This would be much more sophisticated with actual topic models
                consistency = 1.0 / (1.0 + unique_words / 10) if unique_words > 0 else 0
                features['topic_consistency'].append(consistency)
            
            return features
            
        except Exception as e:
            logger.warning(f"Semantic feature extraction failed: {str(e)}")
            return {}

class TextQualityAssessor:
    """Assess and improve text data quality."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
    
    async def assess_text_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Comprehensive text quality assessment."""
        try:
            assessment = {
                'overall_quality': 0.0,
                'quality_scores': [],
                'quality_issues': [],
                'recommendations': [],
                'statistics': {}
            }
            
            # Individual quality scores
            quality_scores = []
            issues = []
            
            for i, text in enumerate(texts):
                score, text_issues = await self._assess_single_text_quality(text)
                quality_scores.append(score)
                
                if text_issues:
                    for issue in text_issues:
                        issues.append({
                            'text_index': i,
                            'issue': issue,
                            'text_preview': text[:100] + '...' if len(text) > 100 else text
                        })
            
            assessment['quality_scores'] = quality_scores
            assessment['quality_issues'] = issues
            assessment['overall_quality'] = np.mean(quality_scores) if quality_scores else 0.0
            
            # Statistics
            assessment['statistics'] = {
                'mean_quality': float(np.mean(quality_scores)) if quality_scores else 0.0,
                'std_quality': float(np.std(quality_scores)) if quality_scores else 0.0,
                'min_quality': float(min(quality_scores)) if quality_scores else 0.0,
                'max_quality': float(max(quality_scores)) if quality_scores else 0.0,
                'low_quality_count': sum(1 for score in quality_scores if score < 0.5),
                'high_quality_count': sum(1 for score in quality_scores if score > 0.8)
            }
            
            # Generate recommendations
            assessment['recommendations'] = await self._generate_quality_recommendations(
                assessment['statistics'], issues
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Text quality assessment failed: {str(e)}")
            return {}
    
    async def _assess_single_text_quality(self, text: str) -> Tuple[float, List[str]]:
        """Assess quality of a single text."""
        try:
            score = 1.0
            issues = []
            
            # Length check
            if len(text) < self.config.min_text_length:
                score *= 0.5
                issues.append("Text too short")
            elif len(text) > self.config.max_text_length:
                score *= 0.8
                issues.append("Text very long")
            
            # Character diversity
            char_diversity = len(set(text.lower())) / len(text) if text else 0
            if char_diversity < 0.1:
                score *= 0.6
                issues.append("Low character diversity")
            
            # Word repetition
            words = text.split()
            if words:
                word_counts = Counter(words)
                max_repetition = max(word_counts.values()) / len(words)
                if max_repetition > 0.3:
                    score *= 0.7
                    issues.append("High word repetition")
            
            # Language coherence (simplified)
            alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
            if alpha_ratio < 0.5:
                score *= 0.8
                issues.append("Low alphabetic character ratio")
            
            # Sentence structure
            sentences = sent_tokenize(text)
            if sentences:
                avg_sent_length = np.mean([len(sent.split()) for sent in sentences])
                if avg_sent_length < 3:
                    score *= 0.7
                    issues.append("Very short sentences")
                elif avg_sent_length > 50:
                    score *= 0.8
                    issues.append("Very long sentences")
            
            # Special character ratio
            special_ratio = sum(1 for c in text if not c.isalnum() and c not in string.whitespace + string.punctuation) / len(text) if text else 0
            if special_ratio > 0.1:
                score *= 0.8
                issues.append("High special character ratio")
            
            return max(0.0, min(1.0, score)), issues
            
        except Exception as e:
            logger.warning(f"Single text quality assessment failed: {str(e)}")
            return 0.5, ["Assessment failed"]
    
    async def _generate_quality_recommendations(
        self,
        statistics: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for improving text quality."""
        try:
            recommendations = []
            
            # Overall quality recommendations
            mean_quality = statistics.get('mean_quality', 0.0)
            if mean_quality < 0.6:
                recommendations.append("Overall text quality is low - consider data cleaning and validation")
            
            # Low quality text handling
            low_quality_count = statistics.get('low_quality_count', 0)
            if low_quality_count > 0:
                recommendations.append(f"Remove or improve {low_quality_count} low-quality texts")
            
            # Issue-specific recommendations
            issue_types = Counter(issue['issue'] for issue in issues)
            
            if 'Text too short' in issue_types:
                count = issue_types['Text too short']
                recommendations.append(f"Consider removing {count} texts that are too short or aggregate them")
            
            if 'High word repetition' in issue_types:
                count = issue_types['High word repetition']
                recommendations.append(f"Review {count} texts with high repetition for duplicates or spam")
            
            if 'Low character diversity' in issue_types:
                count = issue_types['Low character diversity']
                recommendations.append(f"Investigate {count} texts with low diversity - possible encoding issues")
            
            # Variance recommendations
            std_quality = statistics.get('std_quality', 0.0)
            if std_quality > 0.3:
                recommendations.append("High quality variance detected - consider stratified sampling or quality-based filtering")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Quality recommendations generation failed: {str(e)}")
            return ["Review text quality manually"]

class TextDataAugmentation:
    """Augment text data for improved model training."""
    
    def __init__(self, config: TextModelConfig):
        self.config = config
    
    async def augment_texts(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        augmentation_ratio: float = 0.5
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Augment text data using various techniques."""
        try:
            augmented_texts = texts.copy()
            augmented_labels = labels.copy() if labels else None
            
            n_to_augment = int(len(texts) * augmentation_ratio)
            
            # Select texts to augment (prefer minority classes if labels available)
            if labels:
                indices_to_augment = await self._select_texts_for_augmentation(texts, labels, n_to_augment)
            else:
                indices_to_augment = np.random.choice(len(texts), n_to_augment, replace=False)
            
            # Apply augmentation techniques
            for idx in indices_to_augment:
                original_text = texts[idx]
                original_label = labels[idx] if labels else None
                
                # Synonym replacement
                augmented_text = await self._synonym_replacement(original_text)
                if augmented_text != original_text:
                    augmented_texts.append(augmented_text)
                    if augmented_labels:
                        augmented_labels.append(original_label)
                
                # Random insertion
                if np.random.random() < 0.3:
                    augmented_text = await self._random_insertion(original_text)
                    if augmented_text != original_text:
                        augmented_texts.append(augmented_text)
                        if augmented_labels:
                            augmented_labels.append(original_label)
                
                # Random swap
                if np.random.random() < 0.3:
                    augmented_text = await self._random_swap(original_text)
                    if augmented_text != original_text:
                        augmented_texts.append(augmented_text)
                        if augmented_labels:
                            augmented_labels.append(original_label)
            
            logger.info(f"Augmented {len(texts)} texts to {len(augmented_texts)} texts")
            return augmented_texts, augmented_labels
            
        except Exception as e:
            logger.error(f"Text augmentation failed: {str(e)}")
            return texts, labels
    
    async def _select_texts_for_augmentation(
        self,
        texts: List[str],
        labels: List[str],
        n_to_augment: int
    ) -> List[int]:
        """Select texts for augmentation, preferring minority classes."""
        try:
            label_counts = Counter(labels)
            minority_classes = [
                label for label, count in label_counts.items()
                if count < len(labels) / len(label_counts) * 0.8
            ]
            
            # Prefer minority class texts
            minority_indices = [
                i for i, label in enumerate(labels)
                if label in minority_classes
            ]
            
            if len(minority_indices) >= n_to_augment:
                return np.random.choice(minority_indices, n_to_augment, replace=False).tolist()
            else:
                # Include all minority indices and sample from others
                remaining_needed = n_to_augment - len(minority_indices)
                other_indices = [
                    i for i in range(len(labels))
                    if i not in minority_indices
                ]
                
                if remaining_needed > 0 and other_indices:
                    additional_indices = np.random.choice(
                        other_indices, min(remaining_needed, len(other_indices)), replace=False
                    ).tolist()
                    return minority_indices + additional_indices
                else:
                    return minority_indices
                    
        except Exception as e:
            logger.warning(f"Text selection for augmentation failed: {str(e)}")
            return np.random.choice(len(texts), n_to_augment, replace=False).tolist()
    
    async def _synonym_replacement(self, text: str, n_replacements: int = 1) -> str:
        """Replace words with synonyms."""
        try:
            words = word_tokenize(text)
            if len(words) < 2:
                return text
            
            # Simple synonym replacement (would be better with WordNet or word embeddings)
            simple_synonyms = {
                'good': ['great', 'excellent', 'fine', 'nice'],
                'bad': ['terrible', 'awful', 'poor', 'horrible'],
                'big': ['large', 'huge', 'massive', 'enormous'],
                'small': ['tiny', 'little', 'mini', 'compact'],
                'fast': ['quick', 'rapid', 'swift', 'speedy'],
                'slow': ['sluggish', 'gradual', 'delayed', 'leisurely']
            }
            
            modified_words = words.copy()
            replacements_made = 0
            
            for i, word in enumerate(words):
                if replacements_made >= n_replacements:
                    break
                
                word_lower = word.lower()
                if word_lower in simple_synonyms and np.random.random() < 0.3:
                    synonym = np.random.choice(simple_synonyms[word_lower])
                    modified_words[i] = synonym if word.islower() else synonym.capitalize()
                    replacements_made += 1
            
            return ' '.join(modified_words)
            
        except Exception as e:
            logger.warning(f"Synonym replacement failed: {str(e)}")
            return text
    
    async def _random_insertion(self, text: str, n_insertions: int = 1) -> str:
        """Randomly insert words from the text."""
        try:
            words = word_tokenize(text)
            if len(words) < 3:
                return text
            
            for _ in range(n_insertions):
                # Pick a random word from the text
                random_word = np.random.choice(words)
                # Pick a random position to insert
                random_idx = np.random.randint(0, len(words) + 1)
                words.insert(random_idx, random_word)
            
            return ' '.join(words)
            
        except Exception as e:
            logger.warning(f"Random insertion failed: {str(e)}")
            return text
    
    async def _random_swap(self, text: str, n_swaps: int = 1) -> str:
        """Randomly swap words in the text."""
        try:
            words = word_tokenize(text)
            if len(words) < 2:
                return text
            
            for _ in range(n_swaps):
                # Pick two random indices
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                # Swap words
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            return ' '.join(words)
            
        except Exception as e:
            logger.warning(f"Random swap failed: {str(e)}")
            return text

# Advanced business intelligence functions

def calculate_text_business_metrics(
    texts: List[str],
    labels: Optional[List[str]] = None,
    sentiment_analysis: Optional[Dict[str, Any]] = None,
    topic_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate business-relevant metrics from text analysis."""
    try:
        metrics = {
            'content_volume': len(texts),
            'avg_content_length': np.mean([len(text) for text in texts]),
            'content_diversity': len(set(texts)) / len(texts) if texts else 0
        }
        
        # Sentiment business impact
        if sentiment_analysis and 'distribution' in sentiment_analysis:
            sentiment_dist = sentiment_analysis['distribution']
            total = sum(sentiment_dist.values())
            
            if total > 0:
                metrics['sentiment_metrics'] = {
                    'positive_ratio': sentiment_dist.get('positive', 0) / total,
                    'negative_ratio': sentiment_dist.get('negative', 0) / total,
                    'customer_satisfaction_proxy': sentiment_dist.get('positive', 0) / total,
                    'brand_risk_indicator': sentiment_dist.get('negative', 0) / total
                }
        
        # Topic-based business insights
        if topic_analysis and 'analysis' in topic_analysis:
            n_topics = len(topic_analysis['analysis'].get('topic_counts', {}))
            metrics['content_complexity'] = {
                'topic_diversity': n_topics,
                'content_focus': 1 / n_topics if n_topics > 0 else 0,
                'content_strategy_alignment': 'high' if n_topics <= 5 else 'medium' if n_topics <= 10 else 'low'
            }
        
        # Classification business value
        if labels:
            label_dist = Counter(labels)
            metrics['classification_metrics'] = {
                'class_distribution': dict(label_dist),
                'dominant_class': label_dist.most_common(1)[0] if label_dist else None,
                'class_balance_score': 1 - (max(label_dist.values()) - min(label_dist.values())) / sum(label_dist.values()) if label_dist else 0
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Business metrics calculation failed: {str(e)}")
        return {}

def estimate_text_processing_costs(
    n_texts: int,
    avg_text_length: float,
    processing_complexity: str = 'medium',
    cloud_pricing: bool = True
) -> Dict[str, Any]:
    """Estimate costs for text processing operations."""
    try:
        costs = {}
        
        # Base processing cost per character (simplified)
        if processing_complexity == 'low':
            cost_per_char = 0.000001  # $1 per million characters
        elif processing_complexity == 'medium':
            cost_per_char = 0.000005  # $5 per million characters
        else:  # high
            cost_per_char = 0.000020  # $20 per million characters
        
        total_chars = n_texts * avg_text_length
        
        costs['processing_costs'] = {
            'base_processing': total_chars * cost_per_char,
            'storage_cost': total_chars * 0.000001,  # Storage cost
            'api_calls': n_texts * 0.001 if cloud_pricing else 0,  # API call costs
            'total_estimated_cost': total_chars * cost_per_char * 1.2  # 20% overhead
        }
        
        # Time estimates
        if processing_complexity == 'low':
            processing_time_per_text = 0.1  # seconds
        elif processing_complexity == 'medium':
            processing_time_per_text = 1.0  # seconds
        else:  # high
            processing_time_per_text = 5.0  # seconds
        
        costs['time_estimates'] = {
            'total_processing_time_hours': (n_texts * processing_time_per_text) / 3600,
            'estimated_completion_time': f"{(n_texts * processing_time_per_text) / 60:.1f} minutes"
        }
        
        return costs
        
    except Exception as e:
        logger.error(f"Cost estimation failed: {str(e)}")
        return {}

def generate_text_analysis_summary(report: TextAnalysisReport) -> Dict[str, Any]:
    """Generate executive summary of text analysis results."""
    try:
        summary = {
            'executive_summary': [],
            'key_findings': [],
            'actionable_insights': [],
            'business_impact': {},
            'next_steps': []
        }
        
        # Executive summary
        n_docs = report.dataset_info.get('n_documents', 0)
        task_type = report.task_type.value
        
        summary['executive_summary'].append(
            f"Analyzed {n_docs:,} text documents using {task_type} approach"
        )
        
        if report.best_model_result:
            best_score = report.best_model_result.test_score
            model_type = report.best_model_result.model_type.value
            summary['executive_summary'].append(
                f"Achieved {best_score:.1%} accuracy using {model_type} model"
            )
        
        # Key findings from various analyses
        if report.sentiment_analysis and 'distribution' in report.sentiment_analysis:
            sentiment_dist = report.sentiment_analysis['distribution']
            dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get)
            summary['key_findings'].append(
                f"Dominant sentiment: {dominant_sentiment} ({sentiment_dist[dominant_sentiment]} documents)"
            )
        
        if report.topic_analysis and 'analysis' in report.topic_analysis:
            topic_counts = report.topic_analysis['analysis'].get('topic_counts', {})
            if topic_counts:
                best_model = report.topic_analysis['analysis'].get('best_model')
                n_topics = topic_counts.get(best_model, 0)
                summary['key_findings'].append(
                    f"Identified {n_topics} distinct topics in the content"
                )
        
        # Business insights
        summary['actionable_insights'] = report.business_insights[:5]
        
        # Business impact estimation
        if 'vocabulary' in report.text_statistics:
            vocab_size = report.text_statistics['vocabulary']['unique_words']
            summary['business_impact']['content_richness'] = (
                'High' if vocab_size > 10000 else 'Medium' if vocab_size > 1000 else 'Low'
            )
        
        # Next steps
        summary['next_steps'] = report.recommendations[:3]
        
        return summary
        
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return {}

# Export main classes and functions
__all__ = [
    'TextAnalysisEngine',
    'TextModelConfig',
    'TextAnalysisReport',
    'TextModelResult',
    'TextPreprocessor',
    'TextVectorizer',
    'TextClassifier',
    'TopicModeler',
    'AdvancedTextFeatureExtractor',
    'TextQualityAssessor',
    'TextDataAugmentation',
    'create_text_analyzer',
    'quick_text_analysis',
    'get_available_text_models',
    'get_text_analysis_recommendations',
    'calculate_text_business_metrics',
    'estimate_text_processing_costs',
    'generate_text_analysis_summary'
]
