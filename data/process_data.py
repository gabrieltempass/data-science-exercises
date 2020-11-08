import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_file_path, categories_file_path):
    """Load the data from the messages and categories CSV files and concatenate them into a data frame.

    Parameters:
        messages_file_path (str): The path of the CSV file that contains the messages.
        categories_file_path (str): The path of the CSV file that contains the categories.

    Returns:
        df (pandas.core.frame.DataFrame): Concatenated data frame from messages and categories.

    Example:
        df = load_data('disaster_messages.csv', 'disaster_categories.csv')
    """

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    df = messages.join(categories.set_index('id'), on='id')

    return df


def clean_data(df):
    """Split the categories column into multiple ones and drop duplicates from the data frame.

    Parameters:
        df (pandas.core.frame.DataFrame): The data frame to be cleaned.

    Returns:
        df (pandas.core.frame.DataFrame): Cleaned data frame.
    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, ]
    category_col_names = list(map(lambda x: x[0:len(x) - 2], row))
    categories.columns = category_col_names

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('string').str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_file_path):
    """Save the data frame into a SQL database file in the specified path.

    Parameters:
        df (pandas.core.frame.DataFrame): The data frame to be saved as a SQL database file.
        database_file_path (str): The path to save the SQL database file.

    Returns:
        None

    Example:
        save_data(df, 'disaster_response.db')
    """

    database_file_name = os.path.basename(database_file_path)
    database_file_name = os.path.splitext(database_file_name)[0]

    engine = create_engine('sqlite:///' + database_file_path)
    df.to_sql(database_file_name, engine, index=False, if_exists='replace')


def main():
    """Load the messages and categories CSV files, clean the data and save it in as a SQL database.

    Call the following three functions, in sequence:
        1. load_data()
        2. clean_data()
        3. save_data()

    When process_data.py is executed, three parameters must be provided:

    Example:
        python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db
    """

    if len(sys.argv) == 4:

        messages_file_path, categories_file_path, database_file_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_file_path, categories_file_path))
        df = load_data(messages_file_path, categories_file_path)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_file_path))
        save_data(df, database_file_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the file paths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the file path of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response.db')


if __name__ == '__main__':
    main()
