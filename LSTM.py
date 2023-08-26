# -*- coding: utf-8 -*-
"""
LSTM Model by Alan Lopez based on Amazon data All_Beauty. All_Beauty is a dataset 
given by amazon for free based on reviews left on Amazon products.

"""

import json
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('All_Beauty.json', 'r') as f:
    first_line = f.readline()
    print(first_line)
    data = [json.loads(line) for line in f]

# Extract review texts and overall ratings
reviews = []
ratings = []

for item in data:
    if 'reviewText' in item and 'overall' in item:
        reviews.append(item['reviewText'])
        ratings.append(item['overall'])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(sequences, maxlen=100)

# One-hot encode ratings
Y = np.array(ratings) - 1  # subtract 1 to make ratings 0-based
Y = to_categorical(Y)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Define model
model = Sequential()
model.add(Embedding(2000, 200))
model.add(LSTM(100, activation='tanh', return_sequences=True)) 
model.add(Dropout(0.3))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.4))  
model.add(Dense(5, activation='softmax'))  # 5 classes for 5 star ratings

# Compile and train model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
history = model.fit(X_train, Y_train, batch_size=80, epochs=40, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate model
print("\n test accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
