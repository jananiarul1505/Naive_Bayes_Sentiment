# Synthetic Sentiment Analysis using Multinomial & Bernoulli Naive Bayes

This project performs sentiment classification using two Naive Bayes models:
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

A small synthetic dataset of positive and negative reviews is created, vectorized using
CountVectorizer, trained on both models, and compared using accuracy, precision,
recall, and F1-score. Visualizations include performance comparison and top predictive words.

# Features
- Synthetic positive & negative reviews
- Bag-of-Words vectorization
- MultinomialNB vs BernoulliNB comparison
- Model performance metrics
- Top important words visualization for each model

# Dataset
Positive Reviews (sample):
- "This product is excellent!"
- "I love this movie. It is fantastic!"
- "Highly recommend this place."

Negative Reviews (sample):
- "This is a terrible service."
- "I hate this restaurant."
- "A complete waste of money."

# Workflow
1. Create positive and negative review lists
2. Vectorize text using CountVectorizer
3. Train MultinomialNB and BernoulliNB
4. Evaluate both models on:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
5. Visualize:
   - Performance comparison bar chart
   - Top 10 positive/negative words for both models

# How to Run
pip install scikit-learn pandas matplotlib seaborn numpy

python your_script_name.py

# Output Includes
- Feature matrix shape
- Performance scores for both models
- Comparison charts
- Top influential positive/negative words for:
  - MultinomialNB
  - BernoulliNB

# Project Purpose
This project demonstrates how different Naive Bayes classifiers behave when applied to text sentiment analysis, helping beginners understand the difference between frequency-based and binary-based text modeling.
