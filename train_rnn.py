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
import argparse
import os
    

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


def train(model, device, loss_function, optimizer, train_loader, test_loader, epochs):
    
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
    def __init__(self, device: str, target_classes: list, embed_len: int, vocab, hidden_dim: int, n_layers: int, dropout = float):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, 
                          batch_first=True, nonlinearity="relu", dropout=dropout)
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


def main(args: argparse.ArgumentParser) -> None:
    
    
    tokenizer = get_tokenizer("basic_english")
    
    train_file_path = os.path.join(args.data_dir, 'train.csv')
    test_file_path = os.path.join(args.data_dir, 'test.csv')
    
    train_loader, test_loader, vocab = get_data_loaders(train_file_path=train_file_path, 
                                                        test_file_path=train_file_path, 
                                                        batch_size=args.batch_size, 
                                                        max_words=args.max_words)    
    
    
    
    target_classes = list(range(args.num_classes))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_fn = nn.CrossEntropyLoss()
    rnn_classifier = RNNClassifier(device = device, 
                                   target_classes = target_classes, 
                                   embed_len = args.embed_len, 
                                   vocab = vocab,
                                   hidden_dim = args.hidden_dim,
                                   n_layers = args.n_layers,
                                   dropout = args.dropout).to(device)
    
    optimizer = AdamW(rnn_classifier.parameters(), lr=args.lr)
    
    rnn_classifier = train(rnn_classifier, device, loss_fn, optimizer, train_loader, test_loader, args.epochs)
    
    model_save_path = os.path.join(args.model_dir, 'model.pth')
    
    torch.save(rnn_classifier.to(torch.device('cpu')).state_dict(), model_save_path)
    
    return None
    

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--batch-size',
        type = int,
        default = 1024,
        help = 'The batch size for the model.')
    
    parser.add_argument(
        '--embed-len',
        type = int,
        default = 32,
        help = 'The length of the embedding used by the RNN')
    
    parser.add_argument(
        '--hidden-dim',
        type = int,
        default = 50,
        help = 'The number of nodes in the hidden layer of the RNN network.')
    
    parser.add_argument(
        '--n-layers',
        type = int,
        default = 3,
        help = 'The number of hidden layers of the RNN.')
    
    parser.add_argument(
        '--dropout',
        type = float,
        default = 0.2,
        help = 'The network dropout probabilty of each hidden layer.')
    
    parser.add_argument(
        '--num_classes',
        type = int,
        default = 57,
        help = 'The number of output classes for the RNN. Default will work for segment UNSPSC classification but will fail in all other cases.')
    
    parser.add_argument(
        '--max-words',
        type = int,
        default = 64,
        help = 'The maximum length of strings to be considered by the RNN. Any words beyond the length of the max-words value will be truncated.')
    
    parser.add_argument(
        '--lr',
        type = float,
        default = 1e-3,
        help = 'The learning rate of the RNN. ')
    
    parser.add_argument(
        '--epochs',
        type = int,
        default = 200,
        help = 'The number of epochs over which to train the RNN.')
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args = parser.parse_args()
    
    main(args)
    
    

