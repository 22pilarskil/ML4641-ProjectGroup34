import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl

# temp solution to disable SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


# Ensure that nltk resources are downloaded (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text, remove_stopwords=False, lemmatize=False):
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def process_dataset(file_path, remove_stopwords=False, lemmatize=False):
    # Load dataset
    df = pd.read_csv('data/all-data.csv')
    
    df['processed_text'] = df['headline'].apply(lambda x: preprocess_text(x, remove_stopwords, lemmatize))
    
    return df

# Path to the sentiment analysis dataset in the 'data' directory
dataset_path = 'data/all-data.csv'  

# Process the dataset in three different ways
df_lemmatized_no_stopwords = process_dataset(dataset_path, remove_stopwords=True, lemmatize=True)
df_no_lemmatization_with_stopwords = process_dataset(dataset_path)
df_lemmatized_with_stopwords = process_dataset(dataset_path, lemmatize=True)

# Save processed datasets in the 'outputs' directory
df_lemmatized_no_stopwords.to_csv('outputs/dataset_lemmatized_no_stopwords.csv', index=False)
df_no_lemmatization_with_stopwords.to_csv('outputs/dataset_with_stopwords_no_lemmatization.csv', index=False)
df_lemmatized_with_stopwords.to_csv('outputs/dataset_lemmatized_with_stopwords.csv', index=False)
