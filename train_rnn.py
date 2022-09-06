import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW

    

def get_data_loaders(train_file_path: str, test_file_path: str, batch_size: int, max_words: int = 64, string_column: str = 'description'):
    train = (pd.read_csv(train_file_path))
    test = (pd.read_csv(test_file_path))
    
    vocab = build_vocab_from_train_test_data([train,test], string_column)
    
    train = (train
             .assign(tokens = lambda df: (df
                                          [string_column]
                                          .astype(str)
                                          .apply(tokenizer)),
                     vocab = lambda df: (df
                                         ['tokens']
                                         .apply(vocab))))
    
    test = (test
            .assign(tokens = lambda df: (df
                                         [string_column]
                                         .astype(str)
                                         .apply(tokenizer)),
                    vocab = lambda df: (df
                                        ['tokens']
                                        .apply(vocab))))
                
    data_test = to_map_style_dataset(test[['label','vocab']].to_records(index=False))
    data_train = to_map_style_dataset(train[['label','vocab']].to_records(index=False))
    
    
    
    def vectorize_batch(batch):
        Y, X = list(zip(*batch))
        X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X] ## Bringing all samples to max_words length.
        
        return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)
    
    
    train_loader = DataLoader(data_train, batch_size=1024, collate_fn=vectorize_batch, shuffle=True)
    test_loader  = DataLoader(data_test , batch_size=1024, collate_fn=vectorize_batch)
    
    
    return train_loader, test_loader, vocab


def train(model, device, loss_function, optimizer, train_loader, test_loader):
    
    
    for epoch in range(1, epochs + 1):
        print(f'current epoch: {epoch}')
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_inputs = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(device)
            
            model.zero_grad()

            outputs = model(b_inputs)
            
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

    model = test(model, test_loader, device)
    
    return model

def test(model, test_loader, device):
    model.eval()
    correct_total = 0
    bal_acc_tot = 0

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            b_inputs = batch[0].to(device)
            b_labels = batch[1].to(device)

            outputs = model(b_inputs)
            outputs = outputs.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            correct = number_correct(outputs, label_ids)
            correct_total += correct
            bal_acc = balanced_accuracy_score(label_ids.flatten(), np.argmax(outputs, axis=1).flatten())
            bal_acc_tot += bal_acc
            
            if step % 10 == 0:
                print(f'Test step: {step}, Accuracy: {correct/len(batch[0])}, Balanced Accuracy: {bal_acc}')

    print(f"Test set: Accuracy:  {correct_total/len(test_loader.dataset)} Balanced Accuracy: {bal_acc_tot/len(test_loader)}")
    
    return model

class RNNClassifier(nn.Module):
    def __init__(self, device: str, target_classes: list, embed_len: int, vocab, hidden_dim: int, n_layers: int):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, 
                          batch_first=True, nonlinearity="relu", dropout=0.2)
        self.linear = nn.Linear(hidden_dim, len(target_classes))
        self.device = device

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim).to(self.device))
        return self.linear(output[:,-1])    
    
    
def build_vocab_from_train_test_data(dataset_list: list, string_column: str):
    
    strings = ''
    
    for dataset in dataset_list:
        for string in dataset[string_column]:
            strings += string
    
    strings = strings.split()

    vocab = build_vocab_from_iterator([strings], min_freq=1, specials=["<UNK>"])
    
    vocab.set_default_index(vocab["<UNK>"])
    
    return vocab

def number_correct(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


if __name__ =='__main__':
    
    tokenizer = get_tokenizer("basic_english")
    
    train_file_path = 'prepared_data/super_train.csv'
    test_file_path = 'prepared_data/super_test.csv'
    
    batch_size = 1024
    max_words = 64
    
    train_loader, test_loader, vocab = get_data_loaders(train_file_path=train_file_path, 
                                                        test_file_path=train_file_path, 
                                                        batch_size=batch_size, 
                                                        max_words=max_words)    
    
    
    embed_len = 32
    hidden_dim = 50
    n_layers=3
    
    num_classes = 57
    
    target_classes = list(range(num_classes))
    
    epochs = 200
    learning_rate = 1e-3
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_fn = nn.CrossEntropyLoss()
    rnn_classifier = RNNClassifier(device = device, 
                                   target_classes = target_classes, 
                                   embed_len = embed_len, 
                                   vocab = vocab,
                                   hidden_dim = hidden_dim,
                                   n_layers = n_layers).to(device)
    
    optimizer = AdamW(rnn_classifier.parameters(), lr=learning_rate)
    
    rnn_classifier = train(rnn_classifier, device, loss_fn, optimizer, train_loader, test_loader)



