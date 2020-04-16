from collections import defaultdict
from computePRF import computePRF
import pandas as pd
import random

def readTrain(trfile):
    sent_labels = defaultdict(int)
    train = pd.read_csv(trfile,
                       sep=',',
                       header=0,
                       encoding='latin')

    neg_label = 0
    pos_label = 0
    maj = 0
    x = []
    y = []
    for i,label in enumerate(train['label']):
        x.append(train['text'][i])
        y.append(label)
        if label == -1:
            neg_label += 1
        elif label == 1:
            pos_label += 1

    if pos_label > neg_label:
        maj = 1
    print(pos_label,neg_label)
    return maj,x,y


def model(majority,X_train,y_train,X_val,y_val,X_text,y_test,pos_words, neg_words):
    '''
    Baseline
    '''
    train_pred = []
    val_pred = []
    test_pred = []
    for i in range(len(X_train)):
        lst = X_train[i].split(' ')
        pos = neg = 0
        for word in lst:
            word = word.strip()
            if word in pos_words:
                pos += 1
            elif word in neg_words:
                neg += 1
        if pos > neg:
            train_pred.append(1)
        elif pos < neg:
            train_pred.append(-1)
        else:
            train_pred.append(0)
    
    for i in range(len(X_val)):
        lst = X_val[i].split(' ')
        pos = neg = 0
        for word in lst:
            word = word.strip()
            if word in pos_words:
                pos += 1
            elif word in neg_words:
                neg += 1
        if pos > neg:
            val_pred.append(1)
        elif pos < neg:
            val_pred.append(-1)
        else:
            val_pred.append(0)

    for i in range(len(X_test)):
        lst = X_test[i].split(' ')
        pos = neg = 0
        for word in lst:
            word = word.strip()
            if word in pos_words:
                pos += 1
            elif word in neg_words:
                neg += 1
        if pos > neg:
            test_pred.append(1)
        elif pos < neg:
            test_pred.append(-1)
        else:
            test_pred.append(0)
    
    return train_pred,val_pred,test_pred


def random_model(majority,X_train,y_train,X_val,y_val,X_text,y_test,pos_words, neg_words):
    '''
    Random Baseline
    '''
    train_rand_pred = []
    val_rand_pred = []
    test_rand_pred = []
    for i in range(len(X_train)):
        train_rand_pred.append(random.randint(0,2)-1)
    for i in range(len(X_val)):
        val_rand_pred.append(random.randint(0,2)-1)
    for i in range(len(X_test)):
        test_rand_pred.append(random.randint(0,2)-1)
    return train_rand_pred,val_rand_pred,test_rand_pred

def readVocab(vocabfile):
    '''
    Read the positive words and negative words
    '''
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
    majority, X_train,y_train = readTrain(trfile)
    temp, X_val,y_val = readTrain(valfile)
    temp2,X_test,y_test = readTrain(testfile)
    print(majority)
    
    #Random baseline model prediction
    print("Random Baseline")
    train_rand_pred,val_rand_pred,test_rand_pred = random_model(majority,X_train,y_train,X_val,y_val,X_test,y_test,pos_words, neg_words)
    print("train")
    computePRF(y_train,train_rand_pred)
    print("val")
    computePRF(y_val,val_rand_pred)
    print("test")
    computePRF(y_test,test_rand_pred)
    
    #baseline model prediction
    print("Stronger Baseline")
    train_pred,val_pred,test_pred = model(majority,X_train,y_train,X_val,y_val,X_test,y_test,pos_words, neg_words)
    print("train")
    computePRF(y_train,train_pred)
    print("val")
    computePRF(y_val,val_pred)
    print("test")
    computePRF(y_test,test_pred)




