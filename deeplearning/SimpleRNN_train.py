import pandas as pd

# Matplot
import matplotlib.pyplot as plt
# %matplotlib inline

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim
import keras
from gensim.models import Word2Vec
from keras.models import Sequential, load_model
from keras.layers import *
import re
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.sequence import pad_sequences
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
#for word, i in tokenizer.word_index.items():
# if word in w2v_model.wv:
# embedding_matrix[i] = w2v_model.wv[word]
tokenizer = Tokenizer()
#tokenizer.fit_on_texts(df_train.text)
vocab_size = 290419  # len(tokenizer.word_index) + 1
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 32  #1024

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')

df = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1", names=DATASET_COLUMNS)

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

df.target = df.target.apply(lambda x: decode_sentiment(x))

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

import re
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    # convert all @username to "AT_USER"
    text = re.sub('@[^\s]+','AT_USER', text)
    # correct all multiple white spaces to a single white space
    text = re.sub('[\s]+', ' ', text)
    # convert "#topic" to just "topic"
    text = re.sub(r'#([^\s]+)', r'\1', text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))

"""## Settings"""
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))


# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

w2v_model = Word2Vec.load("model.w2v")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)

labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)


encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

embedding_matrix = np.load('embedding_matrix.npy')
print(embedding_matrix.shape)

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)
 
model = Sequential()

model.add(embedding_layer)   # Embedding(max_features, embedding_vecor_length, input_length=SEQUENCE_LENGTH))
model.add(Dropout(0.5))       
model.add(SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=20, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)]


import matplotlib.image as mpimg
import random
def shuffle(samples):
    # NOTE: this is pseudocode
    return random.shuffle(samples)

def generator(samples, batch_size=32):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X)
                #filename = './common_filepath/'+batch_sample[0]
                #image = mpimg.imread(filename)
                # Read label (y)
                x = batch_sample[0]
                y = batch_sample[1]
                # Add example to arrays
                X_train.append(x)
                y_train.append(y)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch
            yield X_train, y_train

# Import list of train and validation data (image filenames and image labels)
# Note this is not valid code.
train_samples = [[x_train[i], y_train[i]] for i in range(x_train.shape[0])]
print(len(x_train))
print('train_shape: ',x_train.shape[0])
validation_samples = [[x_test[i], y_test[i]] for i in range(len(x_test))]

# Create generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#######################
# Use generator to train neural network in Keras
#######################

# Fit model using generator
history = model.fit_generator(train_generator,
                    #samples_per_epoch=len(train_samples),
                    steps_per_epoch=200,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=30,
		    #nb_val_samples=len(validation_samples), 
		    nb_epoch=50)

model.save("SimpleRNN_model.h5")
print(history.history['acc'])
print('and')
print(history.history['val_acc'])
#print(history)




#history = model.fit_generator(x_train, y_train,
#                   # batch_size=BATCH_SIZE,
#                    steps_per_epoch=100, 
#		    validation_steps=100,
#		    epochs=EPOCHS,
 #                   validation_split=0.1,
 #                   verbose=1,
 #                   callbacks=callbacks)

#score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
#print()
#print("ACCURACY:",score[1])
#print("LOSS:",score[0])

