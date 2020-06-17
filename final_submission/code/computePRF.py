import pprint
import argparse
import pandas as pd

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)


def readLabels(datafile):

    f = pd.read_csv(datafile)
    labels = f['label']
    return labels


def computePRF(truthlabels, predlabels):
    pos_pos = 0
    neg_pos = 0
    neu_pos = 0
    
    pos_neg = 0
    neg_neg = 0
    neu_neg = 0
    
    pos_neu = 0
    neg_neu = 0
    neu_neu = 0
    
    for t, p in zip(truthlabels, predlabels):
        if t == 1 and p == 1:
            pos_pos += 1
        elif t == -1 and p == 1:
            neg_pos += 1
        elif t == 0 and p == 1:
            neu_pos += 1
                
        elif t == 1 and p == -1:
            pos_neg += 1
        elif t == -1 and p == -1:
            neg_neg += 1
        elif t == 0 and p == -1:
            neu_neg += 1
        
        elif t == 1 and p == 0:
            pos_neu += 1
        elif t == -1 and p == 0:
            neg_neu += 1
        elif t == 0 and p == 0:
            neu_neu += 1

    
    print("pos_pos:{} pos_neg:{} pos_neu:{} neg_pos:{} neg_neg:{} neg_neu:{} neu_pos:{} neu_neg:{} neu_neu:{}".format(pos_pos, pos_neg, pos_neu, neg_pos, neg_neg, neg_neu, neu_pos, neu_neg, neu_neu))
    
    pos_pre = pos_pos/(pos_pos+neg_pos+neu_pos)
    neg_pre = neg_neg/(pos_neg+neg_neg+neu_neg)
    neu_pre = neu_neu/(pos_neu+neg_neu+neu_neu)
    
    pos_recall = pos_pos/(pos_pos+pos_neu+pos_neg)
    neg_recall = neg_neg/(neg_pos+neg_neu+neg_neg)
    neu_recall = neu_neu/(neu_pos+neu_neu+neu_neg)
    
    pos_f1 = 2*pos_pre*pos_recall/(pos_pre+pos_recall)
    neg_f1 = 2*neg_pre*neg_recall/(neg_pre+neg_recall)
    neu_f1 = 2*neu_pre*neu_recall/(neu_pre+neu_recall)
    
    pre_avg = (pos_pre+neg_pre+neu_pre)/3
    recall_avg = (pos_recall+neg_recall+neu_recall)/3
    f1_avg = (pos_f1+neg_f1+neu_f1)/3
    
    col = ['precision', 'recall', 'f1']
    pos = [pos_pre, pos_recall, pos_f1]
    neg = [neg_pre, neg_recall, neg_f1]
    neu = [neu_pre, neu_recall, neu_f1]
    avg = [pre_avg, recall_avg, f1_avg]
    titles = ['', 'positive', 'negative', 'neutral', 'macroavg']
    
    data = [titles] + list(zip(col, pos, neg, neu, avg))

    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line))

#    return prec, recall, f1


def main(args):
    gold = readLabels(args.goldfile)
    pred = readLabels(args.predfile)

    print("Performance")
    computePRF(gold, pred)


if __name__ == '__main__':
    args = parser.parse_args()
#    pp.pprint(args)
    main(args)
