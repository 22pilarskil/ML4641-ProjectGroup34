import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

finance_news = pd.read_csv('data/all-data.csv', encoding = 'UTF8')
##finance_news = finance_news.head(200)

finance_news.head()

X = finance_news['headline'].to_list()
y = finance_news['sentiment'].to_list()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features based on your data
X_tfidf = vectorizer.fit_transform(X)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_tfidf, y)

# Make predictions on the same dataset
predictions = classifier.predict(X_tfidf)

# Calculate and print accuracy
accuracy = accuracy_score(y, predictions)
print(f'Baseline Model Accuracy: {accuracy * 100:.2f}%')
print(f'Baseline Model Classification Report:\n{classification_report(y, predictions)}')

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0:'neutral', 1:'positive',2:'negative'}

sent_val = list()
for x in X:
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]
   
    val = labels[np.argmax(outputs.detach().numpy())]   
    sent_val.append(val)

print(f'Finbert Model Accuracy: {accuracy_score(y, sent_val) * 100:.2f}%')
print(f'Finbert Model Classification Report:\n{classification_report(y, sent_val)}')

# F1 score
