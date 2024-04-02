# Description of Directories and Files
```/data```: Contains data files necessary for training models, as well as some outputs from analyzing our dataset, including confusion and correlation matrices

```/data/FinancialPhraseBank/```: Financial phrase banks for training the Finbert model

```/data/correlations/```: Correlations of financial features for the tickers we analyzed

```/images/```: Contains images for hosting our reports in the repo

```/outputs/```: Contains outputs after performing preprocessing on our dataset, as well as after training our Finbert model

```/scraping/```: Scripts for scraping data off of news websites and making calls to APIs to create our dataset

```/training_results/```: Results after training our models

```/data/all-data.csv```: The old dataset that we attempted to use but didn't go forward with

```/data/tickers_list.pkl```: Pickle file containing list of tickers in our dataset

```/data/FinancialPhraseBank/Sentences_50Agree.txt```: Finbert dataset with more than 50% of agreement

```/data/FinancialPhraseBank/Sentences_66Agree.txt```: Finbert dataset with more than 66% of agreement

```/data/FinancialPhraseBank/Sentences_66Agree.txt```: Finbert dataset with more than 75% of agreement

```/data/FinancialPhraseBank/Sentences_66Agree.txt```: Finbert dataset with full agreement

```/data/correlations/confusion_matrix.npy```: Confusion matrix from dataset

```/data/correlations/correlations_per_ticker.pkl```: Pickle file with correlations between metrics for each ticker

```/data/correlations/nd_correlations.txt```: Correlations for nxd dataset

```/data/correlations/total_correlations.txt```: Overall correlations

```/scraping/construct_dataset_api.py```: Construct dataset by calling Alpha Vantage API

```/scraping/construct_dataset_scraping.py```: Construct dataset by scraping off of news websites

```/training_results/epoch_results.csv```: Training results after each epoch

```/add_info.py```: Adds percent changes to headline CSV

```/calculate_covariance.py```: Outputs correlation matrices

```/classification_labels_dist.py```: Plots distribution of labels in dataset for classification

```/data_loader.py```: Creates pkl for each ticker financial data

```/dataloader.py```: Gets data for training from various sources

```/finbert_training.py```: Trains Finbert model

```/output_tickers_list.py```: Outputs ticker list

```/text_preprocessing.py```: Outputs preprocessed text data
