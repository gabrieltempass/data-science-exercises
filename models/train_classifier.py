import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle


def load_data(database_file_path):
    """Load the data from the messages and categories CSV files and concatenate them into a data frame.

    Parameters:
        database_file_path (str): The path of the SQL database file.

    Returns:
        X (pandas.core.series.Series):
        y (pandas.core.frame.DataFrame):

    Example:
        X, y = load_data('disaster_response.db')
    """

    database_file_name = os.path.basename(database_file_path)
    database_file_name = os.path.splitext(database_file_name)[0]

    engine = create_engine('sqlite:///' + database_file_path)
    df = pd.read_sql_table(database_file_name, engine)

    df = df.drop(columns=['child_alone'])
    df = df[df['related'] != 2]

    X = df['message']
    y = df.iloc[:, 4:].astype(np.uint8)


def tokenize(text):
    """Tokenize and lemmatize text.

    Arguments:
        text (str): Text message which needs to be tokenized.

    Returns:
        clean_tokens (list): List of tokens extracted from the provided text.
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def evaluate_model(model, X_test, Y_test, category_names):
    print(classification_report(y_test.values, y_test_predict, target_names=y.columns.values))


def save_model(model, model_filepath):
    with open('model.pickle', 'wb') as f:
        pickle.dump(cv, f)


def main():
    if len(sys.argv) == 3:
        database_file_path, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_file_path))
        X, Y, category_names = load_data(database_file_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the file path of the disaster messages database '\
              'as the first argument and the file path of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
