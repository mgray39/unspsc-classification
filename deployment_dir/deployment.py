import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BatchEncoding
from sklearn.metrics import balanced_accuracy_score
import unicodedata
from io import StringIO
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#content types

json_content_type = 'application/json'
csv_content_type = 'text/csv'


MAX_LEN = 128

#distilbert tokenizer - distilbert uncased
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



def net():
    
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",  # Using the distilbert model 
        num_labels=57,  # we have 57 at the segment level
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    
    return model


def model_fn(model_dir):
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = net() 
    
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f: 
        model.load_state_dict(torch.load(f))
    
    #push to whatever the hell device you want 
    model.to(device)
    
    #set to evaluate mode
    model.eval()
    
    return model


def input_fn(request_body, content_type):
    
    if content_type == json_content_type:
        
        logging.info(request_body)
        #logging.info(StringIO(request_body).decode())
        
        
        df = (pd.DataFrame([json.loads(request_body)])
               .assign(description = lambda df: string_cleaning(df['description'])))
        
        tensor_dict = BatchEncoding(prepare_string(df.iloc[0,0], tokenizer, MAX_LEN))
        
        
        return tensor_dict
        
    else:
        print('pong....')
        logger.critical(f'unsupported content type: {content_type}')
    

# inference
def predict_fn(input_object, model):
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    input_object = input_object.to(device)
    
    with torch.no_grad():
        prediction = model(input_object['input_ids'], input_object['attention_mask'])
        
        preds = prediction['logits'].detach().numpy()
        
        logits = preds[0]
        pred_class = logits.argmax()
        
    return {'logits': logits, 'pred_class':pred_class}


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


def prepare_string(string, tokenizer, MAX_LEN):
    
    input_ids = list(tokenizer(string)['input_ids'])
    
    input_ids_padded = []
    while len(input_ids) < MAX_LEN:
        input_ids.append(0)
    
    # attention mask is 0 where length is padded, otherwise it is 1
    
    att_mask = [int(id_ > 0) for id_ in input_ids]
    
    # convert to PyTorch data types.
    test_inputs = torch.tensor([input_ids])
    test_masks = torch.tensor([att_mask])
    
    return {'input_ids': test_inputs, 'attention_mask': test_masks}