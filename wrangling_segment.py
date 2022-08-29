import pandas as pd
from sklearn.model_selection import train_test_split
import glob

#file_paths 
unspsc_codes_path = './data/codes/data-unspsc-codes.csv'
california_file_path = './data/california/purchase-order-data-2012-2015-.csv'

# get all csv_files in australia directory
au_file_list = glob.glob('./data/australia/*.csv')

#size of test set
test_fraction = 0.2

#level of unspsc to prepare training data for
code_level = 'family'

#hold the required number of classes for segment and family so you can assert later
code_level_group_count_dict = {'segment': 57,
                               'family': 465}

#based on the code level, set variables for extraction of information from the code information
if code_level == 'segment':
    unspsc_codes_segment_column = ['Segment']
    unspsc_codes_strings_columns_for_concat = ['Family Name', 'Class Name', 'Commodity Name']
    
elif code_level == 'family':
    unspsc_codes_segment_column = ['Family']
    unspsc_codes_strings_columns_for_concat = ['Class Name', 'Commodity Name']

else:
    raise ValueError('code_level is not recognised as either "family" or "segment". Please check the value and retry.')

    
def main() -> (pd.DataFrame, pd.DataFrame):
    """
    Main Function:
    
    Outputs cleaned training and test data frames with required labels to enable processing by subsequent functions.
    
    """
    
    codes_data = prepare_unspsc_codes_data()
    
    codes_data_label_names = list(codes_data['label_name'].unique())
    
    california_data = (prepare_california_data()
                       .query('label_name in @codes_data_label_names'))
    australia_data = (prepare_and_combine_au_files(au_file_list)
                      .query('label_name in @codes_data_label_names'))
    
    
    dataset_df = (pd.concat([codes_data, california_data, australia_data])
                  .assign(label = lambda df: df.groupby(['label_name']).ngroup())
                  .dropna(how ='any')
                  .assign(label_name = lambda df: df['label_name'].astype(int),
                          label = lambda df: df['label'].astype(int)))
    
    dataset_unique_classes = dataset_df['label_name'].nunique()
    
    print(dataset_unique_classes)
    
    print('check if code numbers preserved...')
    assert dataset_unique_classes == code_level_group_count_dict[code_level]
    print('pass')
    
    
    train_df, test_df = train_test_split(dataset_df, test_size=test_fraction, stratify=dataset_df['label'])
    
    return (train_df, test_df)
    

def prepare_unspsc_codes_data(unspsc_codes_path: str = unspsc_codes_path) -> pd.DataFrame:
    """
    Function which reads and cleans the UNSPSC code data to use as a source for training based upon the inputs
    """
    
    
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


def prepare_california_data(california_file_path: str = california_file_path, 
                            code_level: str = code_level) -> pd.DataFrame:
    
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
                               code_level: str = code_level) -> pd.DataFrame:
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
                                 code_level: str = code_level) -> pd.DataFrame:
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
    
    
if __name__ == '__main__':
    
    train_df, test_df = main()
    
    train_df.to_csv('./prepared_data/family_train.csv', encoding ='utf-8',index = False)
    test_df.to_csv('./prepared_data/family_test.csv', encoding = 'utf-8', index = False)
