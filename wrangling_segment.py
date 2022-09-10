import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import unicodedata
import argparse
import requests

hit_dict = {'codes':['https://data.ok.gov/dataset/18a622a6-32d1-48f6-842a-8232bc4ca06c/resource/b92ad3ac-b0f5-4c62-9bd0-eac023cfd083/download/data-unspsc-codes.csv'],
            'california': ['https://data.ca.gov/dataset/ae343670-f827-4bc8-9d44-2af937d60190/resource/bb82edc5-9c78-44e2-8947-68ece26197c5/download/purchase-order-data-2012-2015-.csv'],
            'australia': ['https://data.gov.au/data/dataset/5c7fa69b-b0e9-4553-b8df-2a022dd2e982/resource/561a549b-5a65-450e-86cf-81d392d8fef3/download/20142015fy.csv',
                          'https://data.gov.au/data/dataset/5c7fa69b-b0e9-4553-b8df-2a022dd2e982/resource/21212500-169f-4745-86b3-6ac1c1174151/download/2016-2017-australian-government-contract-data.csv',
                         'https://data.gov.au/data/dataset/5c7fa69b-b0e9-4553-b8df-2a022dd2e982/resource/bc2097b7-8116-4e9d-9953-98813635892a/download/17-18-fy-dataset.csv',
                         ]}



#hold the required number of classes for segment and family so you can assert later
code_level_group_count_dict = {'segment': 57,
                                   'family': 465}

    
def main(args) -> (pd.DataFrame, pd.DataFrame):
    """
    Main Function:
    
    Outputs cleaned training and test data frames with required labels to enable processing by subsequent functions.
    
    """
    
    if not os.path.exists('./prepared_data/'):
        os.mkdir('./prepared_data/')
    
    if args.download:
        download_files_if_not_present(hit_dict)
    
    #file_paths 
    unspsc_codes_path = args.unspsc_codes_path
    california_file_path = args.california_file_path
    
    # get all csv_files in australia directory
    au_file_list = glob.glob(f'{args.au_file_dir}/*.csv')
    
    #size of test set
    test_fraction = args.test_fraction
    
    if args.summarise_class_count ==0:
        summarise_class_count = None
    else:
        summarise_class_count = args.summarise_class_count
    
    #level of unspsc to prepare training data for
    code_level = args.code_level
    
    codes_data = prepare_unspsc_codes_data(unspsc_codes_path, code_level)
    
    #hold the label names for later checking
    codes_data_label_names = list(codes_data['label_name'].unique())
    
    california_data = (prepare_california_data(california_file_path, code_level)
                       .query('label_name in @codes_data_label_names'))
    
    australia_data = (prepare_and_combine_au_files(au_file_list, code_level)
                      .query('label_name in @codes_data_label_names'))
    
    
    dataset_df = (pd.concat([codes_data, california_data, australia_data])
                  .assign(label = lambda df: df.groupby(['label_name']).ngroup(),
                          description = lambda df: string_cleaning(df['description']))
                  .dropna(how ='any')
                  .assign(label_name = lambda df: df['label_name'].astype(int),
                          label = lambda df: df['label'].astype(int)))
    
    test_class_value = code_level_group_count_dict[code_level]
    
    if summarise_class_count is not None:
        dataset_df = (summarise_classes(dataset_df, summarise_class_count, 'label_name')
                      .assign(label = lambda df: df.groupby(['label_name']).ngroup())
                      .dropna(how ='any')
                      .assign(label_name = lambda df: df['label_name'].astype(int),
                              label = lambda df: df['label'].astype(int))
                      .dropna())
        
        test_class_value = summarise_class_count
    
    
    dataset_unique_classes = dataset_df['label_name'].nunique()
    
    print(dataset_unique_classes)
    
    print('check if code numbers preserved...')
    assert dataset_unique_classes == test_class_value
    print('pass')
    
    
    train_df, test_df = train_test_split(dataset_df.dropna(), test_size=test_fraction, stratify=dataset_df['label'])
    
    return (train_df, test_df)
    

def download_files_if_not_present(hit_dict: dict)->None:
    
    if not os.path.exists('./data'):
        os.mkdir(f'./data')
    
    for folder, url_list in hit_dict.items():
        
        if not os.path.exists(f'./data/{folder}'):
            os.mkdir(f'./data/{folder}')
            
        for url in url_list:
            file_name = url.split('/')[-1]
            full_path = f'./data/{folder}/{file_name}'
            print(f'downloading {url} to {full_path}')
            with open(full_path, 'wb') as f:
                f.write(requests.get(url).content)
            print('    done')
            
    return None

    
def prepare_unspsc_codes_data(unspsc_codes_path: str,
                              code_level: str) -> pd.DataFrame:
    """
    Function which reads and cleans the UNSPSC code data to use as a source for training based upon the inputs
    """
    
    #based on the code level, set variables for extraction of information from the code information
    if code_level == 'segment':
        unspsc_codes_segment_column = ['Segment']
        unspsc_codes_strings_columns_for_concat = ['Family Name', 'Class Name', 'Commodity Name']
    
    elif code_level == 'family':
        unspsc_codes_segment_column = ['Family']
        unspsc_codes_strings_columns_for_concat = ['Class Name', 'Commodity Name']

    else:
        raise ValueError('code_level is not recognised as either "family" or "segment". Please check the value and retry.')
    
    df = (pd.read_csv(unspsc_codes_path, encoding = 'latin-1')
          .pipe(concatenate_and_remove, unspsc_codes_strings_columns_for_concat, unspsc_codes_segment_column)
          .rename(columns = {unspsc_codes_segment_column[0]:'label_name', 
                                 'output_field':'description'})
          .assign(label_name = lambda df: df['label_name'].astype(str)))
    
    return df


def concatenate_and_remove(df: pd.DataFrame, 
                           concat_column_list: list, 
                           index_columns: list) -> pd.DataFrame:
    
    #drop columns we don't want
    df = (df[concat_column_list+index_columns]
          .assign(output_field = ''))
    
    for column in concat_column_list:
        df['output_field'] += df[column].str.lower() + ' '
    
    df = df[index_columns + ['output_field']]
    
    return df


def prepare_california_data(california_file_path: str, 
                            code_level: str) -> pd.DataFrame:
    
    columns = ['Normalized UNSPSC', 'Item Name', 'Item Description']
    
    df = (pd.read_csv(california_file_path)
          .dropna(subset=columns, how = 'any')
          [columns]
          .rename(columns = {'Normalized UNSPSC': 'label_name',
                             'Item Name': 'item_name',
                             'Item Description': 'item_description'})
          .assign(description = lambda df: (df['item_name'] + ' ' + df['item_description']).str.lower(),
                  label_name = lambda df: df['label_name'].astype(int).astype(str))
          .drop(columns = ['item_name','item_description'])
          .assign(label_name = lambda df: unspsc_level_selector(df['label_name'], code_level))
          .dropna(subset = ['label_name'])
          .reset_index(drop=True))
    
    return df



def prepare_au_procurement_file(au_procurement_file_path: str, 
                               code_level: str) -> pd.DataFrame:
    """
    First let me say as an Australian: --- bloody hell, Australia! What are you playing at?!
    
    First work out which way we decided to encode the files because for some reason someone decided 'latin-1' was a thing?!
    
    Then work out which naming convention we decided to operate on for the file because reasons.
    
    Then I have to remove html tags because some uncivilised swine just dumped webpage content into the column.
    
    """
    
    try:
        df = pd.read_csv(au_procurement_file_path, encoding = 'utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(au_procurement_file_path, encoding = 'latin-1')
    
    try:
        columns = ['UNSPSC', 'Description']
        df = (df[columns]
              .rename(columns = {'UNSPSC': 'label_name',
                                 'Description': 'description'}))
    except KeyError:
        columns = ['UNSPSC Code', 'Description']
        df = (df[columns]
             .rename(columns = {'UNSPSC Code': 'label_name',
                                 'Description': 'description'}))
    
    df = (df
          .assign(label_name = lambda df: df['label_name'].astype(str),
                  description = lambda df: (df['description']
                                             .str.replace(r'<[^<>]*>', '', regex=True)
                                             .str.lower()
                                             .str.replace(r'/\\', ' ', regex = True)))
          .assign(label_name = lambda df: unspsc_level_selector(df['label_name'], code_level))
          .dropna(subset = ['label_name'])
          .reset_index(drop=True))
    
    return df


def prepare_and_combine_au_files(au_file_path_list: list, 
                                 code_level: str) -> pd.DataFrame:
    """
    Function to read all of the Australian data.
    
    Expects file path list and a code level string. 
    
    Reads each file and appends. Returns concatenated structure of all files
    """
    
    df = None
    
    for file_path in au_file_path_list:
        if df is None:
            df = prepare_au_procurement_file(file_path, code_level)
            continue
        else:
            df = pd.concat([df, prepare_au_procurement_file(file_path, code_level)])
    
    return df


def unspsc_level_selector(code_series: pd.Series, code_level: str) -> pd.Series:
    
    unspsc_length = 8
    
    selection_length_dict = {'segment':2,
                             'family':4}
    
    select = selection_length_dict[code_level]
    pad = unspsc_length - select

    code_series = ((code_series.str[0:select] + ('0' * pad)) 
                   .where(code_series.str[select-2:select]!='00', None))
    
    return code_series


def summarise_classes(df:pd.DataFrame, super_class_count: int, class_name: str = 'label_name') -> pd.DataFrame:
    df['super_class'] = None
    
    unique_classes = df.sort_values(class_name)[class_name].unique()
    
    group_length = len(unique_classes)//(super_class_count-1)
    for i in range(0,super_class_count):
        group_classes = unique_classes[i*group_length:np.min([(i+1)*group_length, len(unique_classes)])]
        df.loc[df[class_name].isin(group_classes), 'super_class'] = i
    
    df = (df
          .drop(columns = ['label_name'])
          .rename(columns = {'super_class':'label_name'}))
    
    return df

    
def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    
    return only_ascii.decode()


def string_cleaning(string_series: pd.Series) -> pd.Series:
    
    clean_series = (string_series
                    .astype(str)
                    .str.replace('[^\w\s]',' ', regex=True)
                    .str.replace('\n', ' ')
                    .str.replace(r'[\s+]', ' ', regex=True)
                    .apply(remove_accents))
    
    return clean_series
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
       '--unspsc-codes-path',
       type = str,
       default = './data/codes/data-unspsc-codes.csv',
       help = 'The file path of the unspsc codes data. Download from "https://data.ok.gov/dataset/18a622a6-32d1-48f6-842a-8232bc4ca06c/resource/b92ad3ac-b0f5-4c62-9bd0-eac023cfd083/download/data-unspsc-codes.csv"')
    
    parser.add_argument(
       '--california-file-path',
       type = str,
       default = './data/california/purchase-order-data-2012-2015-.csv',
       help = 'The file path of the california purchase data with unspsc codes. Download from "https://data.ca.gov/dataset/ae343670-f827-4bc8-9d44-2af937d60190/resource/bb82edc5-9c78-44e2-8947-68ece26197c5/download/purchase-order-data-2012-2015-.csv"')
    
    parser.add_argument(
       '--au-file-dir',
       type = str,
       default = './data/australia/',
       help = 'Path to the directory which contains the 3 australian purchasing datasets. Download from "https://data.gov.au/dataset/ds-dga-5c7fa69b-b0e9-4553-b8df-2a022dd2e982"')
    
    parser.add_argument(
        '--download',
        default = False,
        action = 'store_true',
        help = 'Pass this flag to turn on downloading the dataset. Downloads data to ./data of the working directory.')
    
    parser.add_argument(
        '--test-fraction',
        type = float,
        default = 0.2,
        help = 'The test split of the dataset. Default is 0.2 (20%)')
    
    parser.add_argument(
        '--summarise_class_count',
        type = int,
        default = 0,
        help = 'The number of classes to summarise the provided classes into. This is used for hierarchical cluster fitting which is not completely implemented yet. Passing 0 which is default will result in no summarising of the classes.')
    
    parser.add_argument(
        '--code-level',
        type = str,
        default = 'segment',
        help = "Choice of either 'segment' or 'family' to create the training data.")
    
    parser.add_argument(
        '--train-file-path',
        type = str,
        default = './prepared_data/train.csv',
        help = 'Train file path for file output. Default is in ./prepared_data')
    
    parser.add_argument(
        '--test-file-path', 
        type = str, 
        default = './prepared_data/test.csv',
        help = 'Test file path for file output. Default is in ./prepared_data')
    
    args = parser.parse_args()
    
    
    
    train_df, test_df = main(args)
    
    train_df.to_csv(args.train_file_path, encoding ='utf-8',index = False)
    test_df.to_csv(args.test_file_path, encoding = 'utf-8', index = False)
