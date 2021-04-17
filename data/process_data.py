import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''This function reads message and categories datasets from csv and returns a merged dataframe
    INPUT:
    messages_filepath: name of messages file
    categories_filepath: name of categories file
    RETURN:
    df: messages and categories merged  dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id',how='inner')
    return df

def get_categories_series(raw):
    '''The function takes categories values as an input and returns the parsed features as a pandas series
    INPUT:
    text: String
    database_filename: name of database
    RETURN: pandas.Series
    '''
    raw = raw.split(';')
    parsed = {}
    for i in raw:
        col,val = i.split('-')[0],i.split('-')[1]
        parsed[col] = int(val)
    return pd.Series(parsed)

def clean_data(df):
    '''This function expands the categories feature into separate columns and their respective values and returns the dataframe
    INPUT:
    df: dataframe
    RETURN:
    df: cleaned dataframe
    '''
    categories = df.categories
    row = categories[0]
    category_colnames = re.sub("\d+", "", row)
    category_colnames = category_colnames.replace('-','')
    category_colnames = category_colnames.split(';')
    
    df[category_colnames] = df.categories.apply(get_categories_series)
    df.drop(columns=['categories'],inplace=True)    
    return df


def save_data(df, database_filename):
    '''The function dumps dataframe to a database
    INPUT:
    df: dataframe
    database_filename: name of database
    RETURN: True
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)
    return True  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()