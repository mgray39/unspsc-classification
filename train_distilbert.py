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
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import balanced_accuracy_score

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook

MAX_LEN = 128

#distilbert tokenizer - distilbert uncased
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def number_correct(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def get_train_data_loader(batch_size):
    training_data = pd.read_csv(os.path.join("prepared_data", "train.csv"))
    descriptions = training_data.description.values
    labels = training_data.label.values

    input_ids = []
    for description in descriptions:
        encoded_description = tokenizer.encode(description, add_special_tokens=True, truncation=True, max_length = MAX_LEN)
        input_ids.append(encoded_description)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # attention mask is 0 where length is padded, otherwise it is 1
    attention_masks = []
    # For each description...
    for desc in input_ids:
        att_mask = [int(token_id > 0) for token_id in desc]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    tensor_train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(tensor_train_data)
    train_dataloader = DataLoader(tensor_train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader


def get_test_data_loader(test_batch_size):
    test_data = pd.read_csv(os.path.join("prepared_data", "test.csv"))
    descriptions = test_data.description.values
    labels = test_data.label.values

    input_ids = []
    for description in descriptions:
        encoded_description = tokenizer.encode(description, add_special_tokens=True, truncation = True, max_length = MAX_LEN)
        input_ids.append(encoded_description)

    # pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # attention mask is 0 where length is padded, otherwise it is 1
    attention_masks = []
    # For each description...
    for desc in input_ids:
        att_mask = [int(token_id > 0) for token_id in desc]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    tensor_test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(tensor_test_data)
    test_dataloader = DataLoader(tensor_test_data, sampler=test_sampler, batch_size=test_batch_size)

    return test_dataloader


def net():
    
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",  # Using the distilbert model 
        num_labels=57,  # we have 57 at the segment level
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    
    return model
    

def train(model, device, loss_function, optimizer, epochs, train_loader, test_loader, hook):
    
    if hook:
        hook.set_mode(modes.EVAL)
    
    for epoch in range(1, epochs + 1):
        print(f'current epoch: {epoch}')
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2]
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_function(outputs[0], b_labels)

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            
            if step % 10  == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
                
        #run a test at the end of each epoch to allow evaluation of model loss
        model = test(model, test_loader, device, hook)
    
    return model

def test(model, test_loader, device, hook):
    
    if hook:
        hook.set_mode(modes.EVAL)
    
    model.eval()
    correct_total = 0

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2]
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            correct = number_correct(logits, label_ids)
            correct_total += correct
            bal_acc = balanced_accuracy_score(label_ids, np.argmax(logits, axis=1).flatten())
            bal_acc_tot += bal_acc
            
            if step % 10 == 0:
                print(f'Test step: {step}, Accuracy: {correct/len(batch[0])}, Balanced Accuracy: {bal_acc}')
            
            

    print("Test set: Accuracy: ", correct_total/len(test_loader.dataset))
    
    return model


def main(args):
    
    #get smdebug logging hook
    hook = smd.Hook.create_from_json_file()
    
    #get train loaders
    train_loader = get_train_data_loader(args.data_dir, args.batch_size)
    test_loader = get_test_data_loader(args.data_dir, args.batch_size)
    
    #loss function - custom to allow explicit hook registration with profiler
    loss_function = nn.CrossEntropyLoss()
    
    #initialise network
    model = net()
    
    #AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.eps,  # args.adam_epsilon - default is 1e-8.
    )
    
    hook.register_hook(model)
    hook.register_loss(loss_function)
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using: {device}')
    model = model.to(device)
    model = train(model, device, loss_function, optimizer, args.epochs, train_loader, test_loader, hook)
    
    model_save_path = os.path.join(args.model_dir, 'model.pth')
    
    torch.save(model.to(torch.device('cpu')).state_dict(), model_save_path)
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 11)",
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5, 
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--eps", 
        type=float, 
        default=1e-8, 
        metavar="EPS",
        help="epsilon (default: 1e-8)"
    )
    
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args = parser.parse_args()
    
    main(args)