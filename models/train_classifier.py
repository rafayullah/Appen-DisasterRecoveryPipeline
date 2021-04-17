import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

nltk.download("stopwords")

def load_data(database_filepath):
    '''This function takes a database filepath as an input and returns X (input) and Y (output) dataframes along with column names of Y dataframe
    INPUT:
    database_filepath: string containing path to database file
    RETURNS:
    X: dataframe of input values
    y: dataframe of output values
    category_names: list of categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    df = df.drop(columns=["id","related"])

    X_columns = df.dtypes[df.dtypes=="object"].index
    Y_columns = df.dtypes[df.dtypes!="object"].index
    X = df[X_columns]
    Y = df[Y_columns]
    return X, Y, Y_columns

# index webpage displays cool visuals and receives user input text for model
def tokenize(text):
    '''The function takes a string input and return a list of processed tokens
    INPUT:
    text: string containing input text
    RETURNS:
    lemmed: list of tokens
    '''
    words = re.findall("[a-zA-Z]+", text)
    
    stop_words = set(stopwords.words('english')) 
    words = [w for w in words if not w in stop_words] 
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed] 
    return lemmed


def build_model():
    '''This function reurns an sklearn pipeline model with GridSearch
    INPUT: None
    Output:
    cv: Gridsearch model
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
#         'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
#         'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
#         'clf__estimator__n_estimators': [5,10,50]
#         'clf__estimator__max_depth' : [5,50,100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''This function evaluates the perfoemance of trained model
    INPUT:
    model: model object
    X_text: test data inputs
    Y_test: test data outputs
    category_names: list of categories
    RETURNS: True
    '''
    y_pred = model.predict(X_test["message"])
    print(classification_report(y_pred, Y_test))
    return True


def save_model(model, model_filepath):
    '''This function takes a model and its path as an input and saves in on local disk at provided path
    INPUT:
    model: model object
    model_filepath: complete filename with path
    OUTPUT: True 
    '''
    joblib.dump(model,model_filepath)
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train["message"], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()