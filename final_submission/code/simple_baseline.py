
from collections import defaultdict
import pandas as pd

def readTrain(trfile):
    sent_labels = defaultdict(int)
    train = pd.read_csv(trfile,
                       sep=',',
                       header=0,
                       encoding='latin')

    neg_label = 0
    pos_label = 0
    maj = 0

    for i,label in enumerate(train['label']):
        text = train['text'][i]
        sent_labels[text]=label
        if label == '0':
            neg_label += 1
        else:
            pos_label += 1

    if pos_label > neg_label:
        maj = 1
    # print(pos_label,neg_label)
    return maj, sent_labels


def model(majority, train_labels, val_labels,pos_words, neg_words):
    # predict
    #print(pos_words)








    print("Writing to results")
    # format is: pred text
    sent = []
    pred = []
    for key, label in train_labels.items():
        sent.append(key)
        pred.append(majority)# to be filled

    df = pd.DataFrame({'label': pred,
                       'text': sent})
    df.to_csv('results.csv',index=False)


def readVocab(vocabfile):
    vocab = set()
    with open(vocabfile, 'r') as f:
        for w in f:
            if w.strip() == '':
                continue
            vocab.add(w.strip())
    return vocab




if __name__ == "__main__":
    posfile = "positive-words.txt"
    negfile = "negative-words.txt"
    trfile = "train.csv"
    valfile = "val.csv"
    testfile = "test.csv"
    outputfile = "predict.txt"

    # read words into sets
    pos_words = readVocab(posfile)
    neg_words = readVocab(negfile)
    # read train and val file
    majority, train_label = readTrain(trfile)
    temp, val_label = readTrain(valfile)
    temp2, test_label = readTrain(testfile)
    print(majority)
    # baseline model prediction
    model(majority,train_label,val_label, pos_words, neg_words)


