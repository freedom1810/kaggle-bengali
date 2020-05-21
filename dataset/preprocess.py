import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random


train = pd.read_csv('/media/asilla/dataset/10_ocr/ben/data/train.csv')
new_rows = [i for i in range(len(train))]
random.seed(2020)
random.shuffle(new_rows)
train = train.iloc[new_rows]

train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
# print(train_labels.shape)
train_paths = train['image_id'].values

labels = {}
labels_path = {}

for i, path in enumerate(train_paths):
    label = str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2])

    if label not in labels:
        labels[label] = 1
        labels_path[label] = [path]
    else:
        labels[label] += 1
        labels_path[label].append(path)

train = {'image_id': [], 'grapheme_root': [],'vowel_diacritic': [],'consonant_diacritic': []}

test= {'image_id': [], 'grapheme_root': [],'vowel_diacritic': [],'consonant_diacritic': []}

for i in labels:

    i_ = [int(x) for x in i.split('_')]
    n_dataset = len(labels_path[i])

    x_train = labels_path[i][:int(n_dataset*0.8)]
    train['image_id'] += x_train
    for _ in range(len(x_train)):
        train['grapheme_root'].append(i_[0])
        train['vowel_diacritic'].append(i_[1])
        train['consonant_diacritic'].append(i_[2])

    x_test = labels_path[i][int(n_dataset*0.8):]
    test['image_id'] += x_test
    for _ in range(len(x_test)):
        test['grapheme_root'].append(i_[0])
        test['vowel_diacritic'].append(i_[1])
        test['consonant_diacritic'].append(i_[2])


df_train = pd.DataFrame(train)
print(type(df_train))
new_rows = [i for i in range(len(df_train))]
random.seed(2020)
random.shuffle(new_rows)
df_train = df_train.iloc[new_rows]
df_train.to_csv('/media/asilla/dataset/10_ocr/ben/data/train_split.csv', index = False)
df_test = pd.DataFrame(test)
new_rows = [i for i in range(len(df_test))]
random.seed(2020)
random.shuffle(new_rows)
df_test = df_test.iloc[new_rows]
df_test.to_csv('/media/asilla/dataset/10_ocr/ben/data/test_split.csv', index = False)
    
