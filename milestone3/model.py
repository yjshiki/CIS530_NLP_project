############################################################
# Imports
############################################################
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import itertools
import csv
import pandas as pd
import spacy
from torchtext.vocab import GloVe
import re
from torchtext import data
import pickle
import torch.optim as optim
# Include your imports here, if any are used.



############################################################
# Neural Networks
############################################################

# NLP = spacy.load('en')
MAX_CHARS = 20000
MAX_VOCAB_SIZE = 25000
comment = data.Field(
    tokenize='spacy'
)
label = data.LabelField()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_data(train_file, val_file, test_file):

    train = data.TabularDataset(
        path=train_file, format='csv', skip_header=True,
        fields=[
            ('label', label),
            ('text', comment),
        ])
    val = data.TabularDataset(
        path=val_file, format='csv', skip_header=True,
        fields=[
            ('label', label),
            ('text', comment),
        ])
    test = data.TabularDataset(
        path=test_file, format='csv', skip_header=True,
        fields=[
            ('label', label),
            ('text', comment),
        ])
    comment.build_vocab(train, vectors=GloVe(name='6B', dim=100),
                        max_size=MAX_VOCAB_SIZE,
                        unk_init=torch.Tensor.normal_
                        )
    label.build_vocab(train)
    print(label.vocab.stoi)

    BATCH_SIZE = 64

    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
        device=device,
        sort=False)
    # pickle.dump(train, open("train_iter.p", "wb"))
    # pickle.dump(val, open("val_iter.p", "wb"))
    # pickle.dump(test, open("test_iter.p", "wb"))
    return train_iterator, val_iterator, test_iterator


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

############################################################
# Reference Code
############################################################
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text)
        # print(predictions.shape, batch.label.shape)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def main():
    train_iterator, val_iterator, test_iterator = load_data('train.csv', 'val.csv', 'test.csv')

    INPUT_DIM = len(comment.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [4]
    OUTPUT_DIM = len(label.vocab)
    DROPOUT = 0.5
    PAD_IDX = comment.vocab.stoi[comment.pad_token]
    N_EPOCHS = 5

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    pretrained_embeddings = comment.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = comment.vocab.stoi[comment.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adadelta(model.parameters(), weight_decay=1e-5) # apply l2-reg
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'baseline-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

if __name__ == '__main__':
    main()




