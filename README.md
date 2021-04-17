# Disaster Response Pipeline Project

Twitter, Facebook and other Social media platforms have been used over the past to explore and study human emotions to events such as sentiments on a newly released product, or to foresee the human biases regarding political elections. Following the trend, during events such as disasters, such valuable information can be utilized to effectively assist the organizations to deliver support.

However, in such an event, the information can be overwhelming. This project is aimed to artificially assist the gathered data into useful information to speed up the response times of concerned organizations.


## Requirements
Install required libraries using the requirements file using the following instruction.
```
pip install -r requirements.txt 
```

## Instructions
Execute the following modules sequentially using the commands below:


* Execute ETL pipeline that cleans data and stores it in a database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

* Execute Machine learning pipeline that trains classifier and saves the model in a local directory
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

* Execute the following command to run the application
```
python app/run.py
```
By default the app runs at port 3001 and can be accessed at http://0.0.0.0:3001/



## Fearures:
### ETL:
Appen provides the raw datasets scrapped from multiple social media platforms. For this project, we use the Disaster Recovery data provided from Appen
The 'data/' module in this project:
* Extracts the datasets from the provided CSV files 
* Merges the datasets
* Cleans and Parses the features
* Loads the dataset to SQL Table

### Machine learning:
Machine learning pipeline extracts data from DB and then splits data in training and test sets.
The model is then initialised. The model consists of a pipeline that first transforms data using CountVectorizer and then TfidfTransformer. Classifier is RandomForestClassifier encapsulated in MultiOutputClassifier for multi-class classification. A list of model parameters is defined and GridSearchCV is applied to search for the best parameters for the model.
Model is then trained on training data and evaluated on test data and model is saved.

### Webapp:
The web app's main page displays charts from the dataset
#### From the Training set
From the Training set, the app displays:
* Distribution of message genres present in training data.
![](https://github.com/rafayullah/Appen-DisasterRecoveryPipeline/blob/main/images/DisasterRecovery%20Genre%20Count.png?raw=true)
* Features Correlation Matrix
![](https://github.com/rafayullah/Appen-DisasterRecoveryPipeline/blob/main/images/DisasterRecovery%20Feature%20Correlation.png?raw=true)


## Authors
* [Rafay Ullah Choudhary](https://github.com/rafayullah

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Acknowledgements
* [Appen](https://appen.com) 