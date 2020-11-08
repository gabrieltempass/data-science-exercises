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
    """Load the data from the SQL database file and divide the input and output variables.

    Parameters:
        database_file_path (str): The path of the SQL database file.

    Returns:
        X (pandas.core.series.Series): Input variables for the machine learning model.
        Y (pandas.core.frame.DataFrame): Output variables for the machine learning model.
        category_names (numpy.ndarray): Array of column names from the Y output variables.

    Example:
        X, Y, category_names = load_data('disaster_response.db')
    """

    database_file_name = os.path.basename(database_file_path)
    database_file_name = os.path.splitext(database_file_name)[0]

    engine = create_engine('sqlite:///' + database_file_path)
    df = pd.read_sql_table(database_file_name, engine)

    df = df.drop(columns=['child_alone'])
    df = df[df['related'] != 2]

    X = df['message']
    Y = df.iloc[:, 4:].astype(np.uint8)
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    """Tokenize and lemmatize text.

    Parameters:
        text (str): Text message which needs to be tokenized.

    Returns:
        clean_tokens (list): List of lemmatized tokens extracted from the provided text.
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
    """Build pipeline that tokenize, apply TFIDF and classify with Random Forest for multilabel output.

    Parameters:
        None.

    Returns:
        pipeline (sklearn.pipeline.Pipeline): Pipeline estimator object.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Predict the Y_test matrix, compare with the real Y_test and print the classification report.

    Parameters:
        model (sklearn.pipeline.Pipeline or classifier object): Classifier object to be used to predict the data.
        X_test (pandas.core.series.Series): Test input variables for the machine learning model.
        Y_test (pandas.core.frame.DataFrame): Test output variables for the machine learning model.
        category_names (numpy.ndarray): Array of column names from the Y output variables.

    Returns:
        None.
    """

    Y_test_predict = model.predict(X_test)
    print(classification_report(Y_test.values, Y_test_predict, target_names=category_names))


def save_model(model, model_file_path):
    """Save the trained model in a pickle file.

    Parameters:
        model (sklearn.pipeline.Pipeline or classifier object): Classifier object to be used to predict the data.
        model_file_path (str): The path to save the model pickle file.

    Returns:
        None.

    Example:
        save_model(model, 'model.pickle')
    """

    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Load the SQL database file, build the model, fit, predict, evaluate and save it in a pickle file.

    Call the following four functions, in sequence:
        1. load_data()
        2. build_model()
        3. evaluate_model()
        4. save_model()

    When train_classifier.py is executed, two parameters must be provided:

    Example:
        python train_classifier.py ../data/disaster-response.db model.pickle
    """

    if len(sys.argv) == 3:
        database_file_path, model_file_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_file_path))
        X, Y, category_names = load_data(database_file_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_file_path))
        save_model(model, model_file_path)

        print('Trained model saved!')

    else:
        print('Please provide the file path of the disaster response SQL database ' \
              'as the first argument and the file path of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/disaster-response.db model.pickle')


if __name__ == '__main__':
    main()
