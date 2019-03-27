from __future__ import absolute_import, division, print_function



import tensorflow as tf
from tensorflow import keras
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

import numpy as np

print(tf.__version__)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_data[3]))


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
#model.add(keras.layers.LSTM(16))
#model.add(keras.layers.Conv2D(filters=16,kernel_size=4,activation=tf.nn.sigmoid))
#model.add(keras.layers.Conv1D(16, 5, activation='relu'))

#model.add(keras.layers.GlobalMaxPooling1D())

model.add(keras.layers.Dense(32, activation='sigmoid'))


#model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(16, activation='sigmoid'))

model.add(keras.layers.Dense(1, activation='sigmoid'))
#model.add(keras.layers.Dense(1, activation='softmax'))


#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


#model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


model.summary()


optim = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optim,
              loss='binary_crossentropy',
              metrics=['acc'])


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]



history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=2,
                   batch_size=20,
                   validation_data=(x_val, y_val),
                  verbose=1)


results = model.evaluate(test_data, test_labels)



tf.keras.utils.plot_model(
    model,
    to_file='modelvert.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB'
)


print(results)