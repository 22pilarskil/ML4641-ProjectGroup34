import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
finance_news = pd.read_csv('data/all-data.csv', encoding='UTF8')

# Split data into X and y
X = finance_news['headline'].to_list()
y = finance_news['sentiment'].to_list()

# Count the occurrences of each sentiment
sentiment_counts = finance_news['sentiment'].value_counts()

# Plot distribution of labels
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['blue', 'green', 'red'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_tfidf, y)

# Make predictions with logistic regression
predictions = classifier.predict(X_tfidf)

# Calculate and print accuracy
accuracy = accuracy_score(y, predictions)
print(f'Baseline Model Accuracy: {accuracy * 100:.2f}%')
print(f'Baseline Model Classification Report:\n{classification_report(y, predictions)}')

# Load Finbert model and tokenizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0: 'neutral', 1: 'positive', 2: 'negative'}

# Make predictions with Finbert model
sent_val = []
for x in X:
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]
    val = labels[np.argmax(outputs.detach().numpy())]
    sent_val.append(val)

# Calculate and print Finbert model accuracy
finbert_accuracy = accuracy_score(y, sent_val)
print(f'Finbert Model Accuracy: {finbert_accuracy * 100:.2f}%')
print(f'Finbert Model Classification Report:\n{classification_report(y, sent_val)}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y, sent_val, labels=['neutral', 'positive', 'negative'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['neutral', 'positive', 'negative'], yticklabels=['neutral', 'positive', 'negative'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Finbert Model')
plt.show()
