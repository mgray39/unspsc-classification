import nlpaug.augmenter.word as naw
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw

original_balance_train_dir = './prepared_data/family_train.csv'
original_balance_test_dir = './prepared_data/family_test.csv'

output_rebalance_train_dir = './prepared_data/rebalanced/family_train.csv'
output_rebalance_test_dir = './prepared_data/rebalanced/family_test.csv'

aug = naw.SynonymAug(aug_src='wordnet')

count_per_class_training = 500

def main():
    
    original_train_df = pd.read_csv(original_balance_train_dir)
    
    rebalanced_train_df, unused_train_df = augment_to_rebalance(original_train_df, count_per_class_training)
    
    rebalanced_train_df.to_csv(output_rebalance_train_dir, index = False, encoding = 'utf-8')
    
    pd.concat([pd.read_csv(original_balance_test_dir), unused_train_df]).to_csv(output_rebalance_test_dir, index = False, encoding = 'utf-8')
    


def augment_to_rebalance(df: pd.DataFrame,  threshold: int) -> (pd.DataFrame, pd.DataFrame):
    
    count_df = (df
                [['label','label_name']]
                .groupby(['label'], as_index=False).count()
                .rename(columns = {'label_name': 'count'}))
    
    over_threshold_labels = count_df.query('count>@threshold')['label'].to_list()
    on_taget_labels = count_df.query('count>@threshold')['label'].to_list()
    under_threshold_labels = count_df.query('count<@threshold')['label'].to_list()
    
    #if it has *exactly* the target value... yes, this actually came up
    
    on_target_df = df.query('label in @on_taget_labels')
    
    #over threshold friends
    
    over_threshold_df = df.query('label in @over_threshold_labels')
    
    over_balanced_train_df = None
    over_unused_df = None
    
    for label in over_threshold_labels:
        sub_df = over_threshold_df.query('label == @label')
        
        if over_balanced_train_df is None:
            over_balanced_train_df, over_unused_df = train_test_split(sub_df, train_size = threshold)
        
        else:
            sub_balanced_df, sub_unused_df = train_test_split(sub_df, train_size = threshold)
            over_balanced_train_df = pd.concat([over_balanced_train_df, sub_balanced_df])
            over_unused_df = pd.concat([over_unused_df, sub_unused_df])
    
    
    # under theshold records
    under_threshold_df = df.query('label in @under_threshold_labels')
    
    augmented_under_df = None
    
    for label in under_threshold_labels:
        
        sub_df = under_threshold_df.query('label == @label')
        
        augmented_sub_df = augment_df_to_value(sub_df, string_column = 'description', threshold = threshold, augmenter = aug)
        
        if augmented_under_df is None:
            augmented_under_df = augmented_sub_df
        
        else:
            augmented_under_df = pd.concat([augmented_under_df, augmented_sub_df])
    
    rebalanced_df = pd.concat([augmented_under_df, on_target_df, over_balanced_train_df])
    
    return (rebalanced_df, over_unused_df)


def augment_df_to_value(df: pd.DataFrame, string_column: str, threshold: int, augmenter = aug) -> pd.DataFrame:
    
    length = len(df)
    add_rows = threshold-length
    
    df_to_be_added = (df
                      .sample(add_rows, replace = True)
                      .assign(description = lambda df: augmenter.augment(df[string_column].to_list()))
                      .rename(columns = {'description': string_column}))
        
    
    df = (pd.concat([df, df_to_be_added]))
    
    return df


if __name__ == '__main__':
    
    main()
