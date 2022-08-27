import json
import os
import sys

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

MAX_LEN = 128 
batch_size=14
epochs=2

#distilbert tokenizer - distilbert uncased
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_train_data_loader(batch_size):
    training_data = pd.read_csv(os.path.join("prepared_data", "train.csv"))
    descriptions = training_data.description.values
    labels = training_data.label.values

    input_ids = []
    for description in descriptions:
        encoded_description = tokenizer.encode(description, add_special_tokens=True)
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
        encoded_description = tokenizer.encode(description, add_special_tokens=True)
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
    

def train(model, device, loss_function):
    train_loader = get_train_data_loader(batch_size)
    test_loader = get_test_data_loader(batch_size)

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    )

    for epoch in range(1, epochs + 1):
        print(f'current epoch: {epoch}')
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = loss_function(outputs, b_labels)

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

    test(model, test_loader, device)

def test(model, test_loader, device):
    model.eval()
    _, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy

    print("Test set: Accuracy: ", eval_accuracy/len(test_loader.dataset))

if __name__ == "__main__":
    
    loss_function = nn.CrossEntropyLoss()
    
    model = net()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train(model, device, loss_function)