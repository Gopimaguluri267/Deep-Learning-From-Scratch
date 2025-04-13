# Word2Vec Implementation Overview

## Class Structure: `train_custom_word2vec()`

### 1. Model Configuration
- **Initialization Parameters:**
  - `approach`: CBOW or Skip-gram
  - `embed_dim`: Embedding dimension size
  - `context_window`: Context window size
  - `min_freq`: Minimum word frequency threshold
  - `subsample_threshold`: Threshold for subsampling
- **Device Setup:** Automatic CUDA/MPS/CPU detection

### 2. Data Processing Pipeline
- **Text Cleaning:** Lowercase, tokenization, stopword removal
- **Vocabulary Management:** 
  - Word frequency-based subsampling
  - Word-to-index mapping creation
- **Training Pair Generation:**
  - CBOW: (context words → target word)
  - Skip-gram: (target word → context words)

### 3. Model Architecture
- **Network Structure:**
  ```
  Input Layer (vocab_size) 
  → Embedding Layer (embed_dim) 
  → Output Layer (vocab_size)
  ```
- **Features:**
  - One-hot encoding for input
  - Linear layers without bias
  - Device-agnostic computation

### 4. Training Implementation
- **Training Process:**
  - Batch processing with DataLoader
  - Train/validation split (90/10)
  - Adagrad optimizer
- **Loss Functions:**
  - Cross Entropy
  - Hierarchical Softmax
  - Negative Sampling
- **Monitoring:**
  - Training loss tracking
  - Validation loss tracking
  - Optional verbose output

### 5. Output
- Trained model
- Training history
- Validation history
- Vocabulary mapping

---
---


# Analysis from the Experiments

## 1. Parameter Impact Analysis

### Vector Size Impact
- **100-dimensional** vectors performed best, showing optimal balance between:
  - Capturing semantic relationships
  - Computational efficiency
  - Model convergence (lowest validation loss)
- Larger dimensions (300) led to overfitting
- Smaller dimensions (50) had insufficient representational capacity

### Window Size Effects
- Window size of **5** proved most effective:
  - Captured meaningful contextual relationships
  - Balanced local and broader semantic context
- Larger windows (10) introduced noise
- Smaller windows (3) missed important relationships

### Sampling Techniques
- **Hierarchical Softmax** outperformed negative sampling
- Subsampling threshold of 1e-5 improved quality by:
  - Reducing impact of very frequent words
  - Better preserving semantic relationships
  - Faster training convergence

## 2. CBOW vs Skip-Gram Comparison

### CBOW Strengths
- Better performance on frequent words
- Faster training (fewer updates)
- More stable for our music theory corpus
- Lower final loss values (0.0057 vs 2.6643)

### Skip-Gram Strengths
- Better handling of rare words
- More precise for specific musical terms
- Captured more fine-grained relationships

## 3. Limitations of Current Implementation

### Core Limitations
1. **Polysemy Handling**
   - Cannot handle multiple meanings of same word
   - E.g., "note" as musical note vs written note

2. **Out-of-Vocabulary Issues**
   - No handling of unseen musical terms
   - Limited to training vocabulary

3. **Context Understanding**
   - Fixed window size limits contextual understanding
   - Misses long-range dependencies


## 4. Proposed Improvements

### Technical Enhancements
 **Subword Information**
   - Implement BPE or WordPiece tokenization
   - Better handle technical musical terms


### Architectural Improvements
 **Dynamic Window Sizing**
   - Adapt window size based on sentence structure
   - Better capture musical phrase relationships

