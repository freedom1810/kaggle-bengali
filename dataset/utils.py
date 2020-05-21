import pandas as pd
import cv2
import numpy as np
import time

def load_data():
    #train
    train = pd.read_csv('/media/asilla/data123/sonng/ben/train_split.csv')
    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    tic = time.time()
    train_images = []
    for i, path in enumerate(train['image_id'].values):
        img = cv2.imread('/media/asilla/data123/sonng/train_raw/' + path + '.jpg')
        train_images.append(img)

    #test
    test = pd.read_csv('/media/asilla/data123/sonng/ben/test_split.csv')
    test_labels = test[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values


    test_images = []
    for i, path in enumerate(test['image_id'].values):
        img = cv2.imread('data/train/' + path + '.jpg')
        test_images.append(img)
            
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    return train_images, test_images, train_labels, test_labels


def load_data_trung(train_path = 'data/train_split.csv',
                    test_path = 'data/test_split.csv' ):
    #train
    #print('train_path: ', train_path)
    train = pd.read_csv(train_path)
    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    tic = time.time()
    train_images = []
    
    train_images = train['image_id'].values
    #test
    test = pd.read_csv(test_path)
    test_labels = test[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values


    test_images = []
    
    test_images = test['image_id'].values
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    return train_images, test_images, train_labels, test_labels

def load_data_1295(train_path = '/home/asilla/sonnh/k/data/train_split.csv'):
    #train
    train = pd.read_csv(train_path)
    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    
    train_images = train['image_id'].values

    train_paths = train['image_id'].values

    labels = set()
    for i, path in enumerate(train_paths):
        labels.add(str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2]))

    labels = list(labels)
    final_train_labels = []
    for i, path in enumerate(train_paths):
        label = str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2])

        final_train_labels.append(labels.index(label))

    return train_images, np.array(final_train_labels)


