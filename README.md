# 🎭 Sentiment Analysis with LSTM & TF-IDF

A machine learning project that analyzes text sentiment using LSTM neural networks combined with TF-IDF vectorization, deployed as an interactive web application using Streamlit.

## 📋 Project Overview

This project implements a sentiment analysis system that can classify text into four categories:
- **Positive** 😊
- **Negative** 😞  
- **Neutral** 😐

## 🏗️ Architecture

The model combines two powerful techniques:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts text to numerical features
- **LSTM (Long Short-Term Memory)**: Deep learning model for sequence processing

### Model Pipeline:
1. **Text Preprocessing**: Clean and normalize input text
2. **Feature Extraction**: Convert text to TF-IDF vectors (5000 features)
3. **Neural Network**: LSTM layers process the vectorized text
4. **Classification**: Output probabilities for each sentiment class

## 🛠️ Technical Stack

- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Text Processing**: TF-IDF Vectorization
