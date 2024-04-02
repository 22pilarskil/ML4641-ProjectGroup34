import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_pickle('data/output_dataset.pkl')

def label_percent_change(change):
    if change >= 0.01:
        return 'positive'
    elif change <= -0.01:
        return 'negative'
    else:
        return 'neutral'

dataset['label'] = dataset['percent change'].apply(label_percent_change)

label_counts = dataset['label'].value_counts()
label_counts.plot(kind='bar', color=['red', 'blue', 'green'])

plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Distribution of Labels')

plt.show()




