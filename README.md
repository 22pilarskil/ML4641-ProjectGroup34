## Directory and File Descriptions

`/`: Root directory of the project.

`/.gitignore`: Specifies intentionally untracked files to ignore when using Git.

`/git_add.sh`: Shell script to recursively find all .sh and .py files in the current directory and add them to staging (used to prevent unintentionally adding images or zip files)

`/train.py`: Python script to run the training process for the stock prediction model.

`/add_info.py`: Python script to add percent change information to our headline dataset. Our default dataset format is initially composed of three columns: ticker, date, headline. This script adds a fourth column, percent change, which is our y actual for this model.

`/get_data_stats.py`: Python script to process pickle files and get the max and min values across all DataFrames. Used to visualize data distributions.

`/dataloader.py`: Python script to create data loaders for the model and clean up the CSV file according to preset filters. 

`/run_model.py`: Python script to train the model and get the loss function. Includes two methods, train and evaluate. 

`/graph_results.py`: Python script to plot the training and validation loss, accuracy, and F1 score per epoch given pickled results obtained from running train.py/

`/model.py`: Contains the definition and architecture of the StockPredictor model we are using for this project

`/index.html`: HTML file containing midterm report

`/images/`: Directory containing images for midterm report

`/scraping/find_news_headlines.py`: Script to scrape headlines from Alphavantage for dataset creation