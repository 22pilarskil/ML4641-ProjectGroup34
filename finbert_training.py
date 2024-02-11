import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

finance_news = pd.read_csv('data/all-data.csv', encoding = 'UTF8')
finance_news = finance_news.head(200)

finance_news.head()

X = finance_news['headline'].to_list()
y = finance_news['sentiment'].to_list()

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0:'neutral', 1:'positive',2:'negative'}

sent_val = list()
for x in X:
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]
   
    val = labels[np.argmax(outputs.detach().numpy())]   
    sent_val.append(val)

print(accuracy_score(y, sent_val))
