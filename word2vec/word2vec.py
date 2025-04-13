import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


class train_custom_word2vec():
    """A class to train custom Word2Vec models using either CBOW or Skip-gram approach.
    
    This class implements the Word2Vec algorithm from scratch, supporting both Continuous
    Bag of Words (CBOW) and Skip-gram architectures. It includes preprocessing, vocabulary
    creation, and model training functionality.
    """
    def __init__(self, approach, embed_dim, context_window, min_freq=1, subsample_threshold=0):
        """Initialize the Word2Vec trainer.
        
        Args:
            approach (str): Either 'cbow' or 'skip-gram' for model architecture
            embed_dim (int): Dimension of word embeddings
            context_window (int): Size of context window for word pairs
            min_freq (int): Minimum frequency for words to be included in vocabulary
            subsample_threshold (float): Threshold for subsampling frequent words
        """
        self.word_tokenize = word_tokenize
        self.stopwords = stopwords
        self.approach = approach
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.min_freq = min_freq
        self.subsample_threshold = subsample_threshold

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        pass

    def cleaning_raw_text(self, text):
        """Preprocess raw text by converting to lowercase, removing punctuation and stopwords.
        
        Args:
            text (str): Raw input text
            
        Returns:
            list: List of cleaned words
        """
        text = text.lower()
        words = self.word_tokenize(text)
        words_punct_rem = [word for word in words if word.isalpha()]
        stop_words_en = set(self.stopwords.words('english'))
        words_stopwords_rem = [word for word in words_punct_rem if word not in stop_words_en]
        
        return words_stopwords_rem
    
    def subsample(self, corpus, threshold):
        """Subsample frequent words to reduce training time and improve quality.
        
        Args:
            corpus (list): List of words
            threshold (float): Frequency threshold for subsampling
            
        Returns:
            list: Subsampled corpus
        """
        word_counts = Counter(corpus)
        total_count = len(corpus)
        word_freq = {word: count / total_count for word, count in word_counts.items()}
        discard_probs = {word: 1 - (threshold / freq)**0.5 for word, freq in word_freq.items() if freq > threshold}
        subsampled = [word for word in corpus if word not in discard_probs or random.random() > discard_probs[word]]
        return subsampled
    
    def create_vocab_mapping(self, cleaned_words):
        """Create vocabulary mapping from words to indices.
        
        Args:
            cleaned_words (list): List of preprocessed words
            
        Returns:
            dict: Word to index mapping
        """
        self.vocab_mapping = {}
        unique_words = np.unique(np.array(cleaned_words)).tolist()
        for i, word in enumerate(unique_words):
            if word not in self.vocab_mapping:
                self.vocab_mapping[word] = i
        return self.vocab_mapping

    def create_pairs(self, cleaned_words):
        """Generate training pairs based on the selected approach (CBOW or Skip-gram).
        
        Args:
            cleaned_words (list): List of preprocessed words
            
        Returns:
            list: Training pairs (context-target pairs)
        """
        if self.min_freq > 1:
            cleaned_words = [word for word in cleaned_words if cleaned_words.count(word) >= self.min_freq]
        if self.subsample_threshold > 0:
            cleaned_words = self.subsample(corpus=cleaned_words, threshold=self.subsample_threshold)
        self.create_vocab_mapping(cleaned_words=cleaned_words)
        if self.approach == "cbow":
            cbow_pairs = []
            for i, target_word in enumerate(cleaned_words):
                context_words = cleaned_words[max(0, i-self.context_window):i] + cleaned_words[i+1:min(len(cleaned_words), i+self.context_window+1)]
                cbow_pairs.append((context_words, target_word))
            return cbow_pairs

        elif self.approach == "skip-gram":
            skipgram_pairs = []
            for i, target_word in enumerate(cleaned_words):
                context_words = cleaned_words[max(0, i-self.context_window):i] + cleaned_words[i+1:min(len(cleaned_words), i+self.context_window+1)]
                tmp_pairs = [(target_word, context_word) for context_word in context_words]
                skipgram_pairs.extend(tmp_pairs)
            return skipgram_pairs
    
    def one_hot_encoding(self, word, vocab_size):
        """Convert a word to its one-hot encoded representation.
        
        Args:
            word (str): Input word
            vocab_size (int): Size of vocabulary
            
        Returns:
            torch.Tensor: One-hot encoded vector
        """
        # one_hot_vector = np.zeros(vocab_size)
        one_hot_vector = torch.zeros(vocab_size)
        one_hot_vector[self.vocab_mapping[word]] = 1
        return one_hot_vector
    
    def create_training_data(self, text):
        """Create training data from raw text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            list: Training data as pairs of tensors
        """
        cleaned_words = self.cleaning_raw_text(text=text)
        # self.create_vocab_mapping(cleaned_words=cleaned_words)
        pairs = self.create_pairs(cleaned_words=cleaned_words)
        if self.approach=="cbow":
            training_data = []
            for context_words, target_word in pairs:
                context_words_one_hot = [self.one_hot_encoding(word=word, vocab_size=len(self.vocab_mapping)) for word in context_words]
                target_word_one_hot = self.one_hot_encoding(word=target_word, vocab_size=len(self.vocab_mapping))
                training_data.append((torch.sum(torch.stack(context_words_one_hot), dim=0), target_word_one_hot))
            return training_data

        elif self.approach=="skip-gram": 
            training_data = []
            for target_word, context_word in pairs:
                target_word_one_hot = self.one_hot_encoding(word=target_word, vocab_size=len(self.vocab_mapping))
                context_word_one_hot = self.one_hot_encoding(word=context_word, vocab_size=len(self.vocab_mapping))
                training_data.append((target_word_one_hot, context_word_one_hot))
            return training_data
    
    def initialize_model(self):
        """Initialize the neural network model layers."""
        self.linear1 = nn.Linear(len(self.vocab_mapping), self.embed_dim, bias=False)
        self.linear2 = nn.Linear(self.embed_dim, len(self.vocab_mapping), bias=False)
    
    def create_model(self):
        """Create and move the model to the appropriate device."""
        self.initialize_model()
        self.model = nn.Sequential(self.linear1, self.linear2)
        self.model.to(self.device)
        
    def train_w2v(self, text, num_epochs, learning_rate, batch_size, loss="cross_entropy", verbose=False):
        """Train the Word2Vec model.
        
        Args:
            text (str): Training text
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimization
            batch_size (int): Size of training batches
            loss (str): Loss function to use
            verbose (bool): Whether to print training progress
            
        Returns:
            tuple: (trained model, training losses, validation losses, vocabulary mapping)
        """
        training_data = self.create_training_data(text=text)
        train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        self.create_model()
        if loss == "cross_entropy":
            loss = nn.CrossEntropyLoss()
        elif loss == "heirarchical_softmax":
            loss = nn.BCEWithLogitsLoss()
        elif loss == "negative_sampling":
            loss = nn.NegativeSamplingLoss()
        optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            total_val_loss = 0
            num_train_batches = 0
            num_val_batches = 0

            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                y_train_logits = self.model(inputs)
                train_loss = loss(y_train_logits, labels)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                num_train_batches += 1
            
            average_train_loss = total_train_loss/num_train_batches
            train_losses.append(average_train_loss)
        
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y_val_logits = self.model(inputs)
                    val_loss = loss(y_val_logits, labels)
                    total_val_loss += val_loss.item()
                    num_val_batches += 1
            
            average_val_loss = total_val_loss/num_val_batches
            val_losses.append(average_val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")
        
        return self.model, train_losses, val_losses, self.vocab_mapping


class evaluate_w2v():
    """A class to evaluate trained Word2Vec models.
    
    Provides functionality for finding similar words and performing analogy tests
    using the trained word embeddings.
    """
    def __init__(self, word_vectors, vocab_mapping):
        """Initialize the evaluator.
        
        Args:
            word_vectors (dict): Word to vector mapping
            vocab_mapping (dict): Word to index mapping
        """
        self.word_vectors = word_vectors
        self.vocab_mapping = vocab_mapping
        self.idx2word_mapping = {v: k for k, v in self.vocab_mapping.items()}
        
        
    def find_similar_words(self, target_word, top_n=10):
        """Find most similar words to a target word using cosine similarity.
        
        Args:
            target_word (str): Word to find similar words for
            top_n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        target_vector = self.word_vectors[target_word]
        similarities = cosine_similarity([target_vector], list(self.word_vectors.values()))[0]
        most_similar = similarities.argsort()[:-top_n-1:-1]
        results = []
        for idx in most_similar:
            word = self.idx2word_mapping[idx]
            if word != target_word:
                results.append((word, similarities[idx]))
        return results
    
    def analogy_test(self, word1, word2, word3, top_n=10):
        """Perform word analogy test (e.g., king - man + woman = queen).
        
        Args:
            word1 (str): First word in analogy
            word2 (str): Second word in analogy
            word3 (str): Third word in analogy
            top_n (int): Number of results to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        v1 = self.word_vectors[word1]
        v2 = self.word_vectors[word2]
        v3 = self.word_vectors[word3]
        target_vector = v2 - v1 + v3
        similarities = cosine_similarity([target_vector], list(self.word_vectors.values()))[0]
        most_similar = similarities.argsort()[-top_n:][::-1]
        results = []
        for idx in most_similar:
            word = self.idx2word_mapping[idx]
            if word not in [word1, word2, word3]:
                results.append((word, similarities[idx]))
        return results
    

class visualize_w2v():
    """A class to visualize Word2Vec embeddings using t-SNE."""
    def __init__(self, word_vectors, vocab_mapping):
        """Initialize the visualizer.
        
        Args:
            word_vectors (dict): Word to vector mapping
            vocab_mapping (dict): Word to index mapping
        """
        self.word_vectors = word_vectors
        self.vocab_mapping = vocab_mapping
        self.idx2word_mapping = {v: k for k, v in self.vocab_mapping.items()}
        
    
    def plot_word_embeddings(self, num_words=50, perplexity=5):
        """Visualize word embeddings using t-SNE dimensionality reduction.
        
        Args:
            num_words (int): Number of words to visualize
            perplexity (int): t-SNE perplexity parameter
        """
        words = list(self.word_vectors.keys())
        
        if num_words < len(words):
            selected_words = random.sample(words, num_words)
        else:
            selected_words = words
            
        selected_vectors = []
        for word in selected_words:
            vector = self.word_vectors[word].numpy()
            selected_vectors.append(vector)
        selected_vectors = np.array(selected_vectors)
        
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(selected_words)-1), 
                    random_state=42, init='pca')
        reduced_vectors = tsne.fit_transform(selected_vectors)
        
        plt.figure(figsize=(15, 10))
        
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
        
        for i, word in enumerate(selected_words):
            plt.annotate(word, 
                        (reduced_vectors[i, 0], reduced_vectors[i, 1]),
                        fontsize=8,
                        alpha=0.7)
        
        plt.title("Word Embeddings Visualization (t-SNE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.show()

