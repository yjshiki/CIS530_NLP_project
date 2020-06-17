import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import re
import os
import pickle

t140 = pd.read_csv('training.1600000.processed.noemoticon.csv',
                   sep=',',
                   header=None,
                   encoding='latin')

label_text = t140[[0, 2, 5]]

print("init ", label_text.head(10))

# Convert labels to range 0-1                                        
label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['label', 'date', 'text']

# Assign proper column names to labels
label_text.head()

def process_text(text):
    text.lower()
    # convert all urls to sting "URL"
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    # convert all @username to "AT_USER"
    text = re.sub('@[^\s]+', 'AT_USER', text)
    # correct all multiple white spaces to a single white space
    text = re.sub('[\s]+', ' ', text)
    # convert "#topic" to just "topic"
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


label_text.text = label_text.text.apply(process_text)
print(label_text.head(30))

TRAIN_SIZE = 0.75
VAL_SIZE = 0.05
dataset_count = len(label_text)

df_train_val, df_test = train_test_split(label_text, test_size=1-TRAIN_SIZE-VAL_SIZE, random_state=42)
df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE / (VAL_SIZE + TRAIN_SIZE), random_state=42)

# with open('outfile', 'wb') as fp:
#     pickle.dump(itemlist, fp)
print(type(df_test), type(df_train))
df_train.to_csv('train.csv', index=False)
df_val.to_csv('val.csv', index=False)
df_test.to_csv('test.csv', index=False)

print("TRAIN size:", len(df_train))
print("VAL size:", len(df_val))
print("TEST size:", len(df_test))

