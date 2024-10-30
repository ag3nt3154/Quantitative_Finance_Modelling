import os
import pandas as pd


def load_df(df_path):
    '''
    Load dataframe from csv or excel files
    '''

    file_type = df_path.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(df_path)
    elif file_type == 'xlsx':
        df = pd.read_excel(df_path)
    else:
        assert False, 'WRONG FILE TYPE!'

    try:
        df.drop('Unnamed: 0', axis='columns', inplace=True)
    except KeyError:
        pass
    return df



def create_folder(folder_path):
    '''
    Creates folder at folder_path.
    Does not create anything if folder exists
    '''
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return