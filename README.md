# Emotion Multimodal Fusion Model

A deep learning model for emotion recognition from multimodal data (text, audio, and visual features) achieving a score of 0.606 on the ML Hackathon EC Campus dataset.

##  Project Overview

This project implements a multimodal fusion approach for emotion recognition using text, audio, and visual features extracted from dialogue videos. The model combines BERT embeddings for text processing, computer vision techniques for facial feature extraction, and audio analysis for prosodic features.

##  Problem Statement

The task involves classifying emotions from multimodal dialogue data containing:
- **Text**: Utterances from conversations
- **Audio**: Speech audio segments
- **Visual**: Video frames with facial expressions
- **Metadata**: Dialogue context, timing, and speaker information

##  Architecture

### Multimodal Feature Extraction

1. **Text Features (BERT)**
   - Uses BERT-base-uncased for text embedding
   - Extracts 768-dimensional feature vectors
   - Handles text preprocessing (lowercase, special character removal)

2. **Visual Features (OpenCV)**
   - Face detection using Haar cascades
   - Extracts facial ROI features:
     - Mean, standard deviation, min, max pixel values
     - Edge detection features using Canny algorithm
   - Aggregates frame-level features across video segments

3. **Audio Features (Librosa)**
   - Zero crossing rate
   - Spectral centroid
   - MFCC coefficients (20 features)
   - Statistical features (mean, std, min, max)

4. **Metadata Features**
   - Utterance duration
   - Position in dialogue
   - Relative position in conversation

### Fusion Models

The project implements two fusion strategies:

#### 1. Early Fusion Model
- Concatenates all modality features before classification
- Uses a deep neural network with multiple layers
- Architecture: `[Text + Visual + Metadata] → Hidden Layers → Classification`

#### 2. Late Fusion Model
- Processes each modality separately
- Learns optimal fusion weights for each modality
- Architecture: `Text → Classifier + Visual → Classifier + Metadata → Classifier → Weighted Fusion`

##  Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Early Fusion | ~0.58 | ~0.38 |
| Late Fusion | ~0.58 | ~0.28 |
| **Final Score** | **0.606** | **0.606** |

##  Technical Implementation

### Dependencies
```python
torch
transformers
opencv-python
librosa
av
scikit-learn
pandas
numpy
```

### Key Features

- **Robust Preprocessing**: Handles missing data, text cleaning, and feature normalization
- **Modular Design**: Separate feature extractors for each modality
- **Scalable Architecture**: Supports both early and late fusion approaches
- **GPU Support**: CUDA acceleration for faster training
- **Model Persistence**: Saves trained models and feature scalers

### Data Pipeline

1. **Data Loading**: CSV files with utterance metadata
2. **Feature Extraction**: Parallel processing of text, audio, and video
3. **Feature Standardization**: Z-score normalization per modality
4. **Model Training**: Cross-validation with early stopping
5. **Evaluation**: Accuracy and F1-score metrics

##  Usage

### Environment Setup
```bash
pip install torch transformers opencv-python librosa av scikit-learn pandas numpy
```

### Training the Model
```python
# Load and preprocess data
train, test = preprocess_data(train_df, test_df)

# Extract multimodal features
train_features, train_indices = extract_features_from_dataset(train, video_dir, is_training=True)
test_features, test_indices = extract_features_from_dataset(test, video_dir, is_training=False)

# Train fusion models
early_fusion = EarlyFusionModel(...)
late_fusion = LateFusionModel(...)

# Evaluate and compare
early_acc, early_f1 = train_and_evaluate("Early Fusion", early_fusion, ...)
late_acc, late_f1 = train_and_evaluate("Late Fusion", late_fusion, ...)
```

### Making Predictions
```python
# Load trained model
model = LateFusionModel(...)
model.load_state_dict(torch.load('final_late_fusion_model.pth'))

# Extract features for new data
features = extract_features_from_dataset(new_data, video_dir, is_training=False)

# Make predictions
predictions = model(features['text'], features['visual'], features['metadata'])
```

##  Key Insights

1. **Modality Importance**: Text features showed the strongest performance, followed by visual and audio features
2. **Fusion Strategy**: Early fusion performed better than late fusion for this dataset
3. **Feature Engineering**: Metadata features (dialogue context) significantly improved performance
4. **Data Quality**: Robust preprocessing was crucial for handling noisy multimodal data

##  Future Improvements

- **Advanced Architectures**: Implement attention mechanisms and transformer-based fusion
- **Data Augmentation**: Synthetic data generation for underrepresented emotions
- **Ensemble Methods**: Combine multiple fusion strategies
- **Hyperparameter Tuning**: Bayesian optimization for optimal model configuration
- **Real-time Processing**: Optimize for real-time emotion recognition

##  Performance Analysis

The model achieved a competitive score of 0.606, demonstrating the effectiveness of multimodal fusion for emotion recognition. The combination of text semantics, facial expressions, and prosodic features provides a comprehensive understanding of emotional states in dialogue contexts.

##  Contributing

This project was developed for the ML Hackathon EC Campus competition. For questions or contributions, please refer to the original competition guidelines.

---

**Note**: This implementation focuses on robust feature extraction and effective fusion strategies for multimodal emotion recognition, achieving competitive performance on the given dataset.