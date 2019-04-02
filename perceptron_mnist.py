import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.io as sio
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

##SIMPLE PERCEPTRON##

"""
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
y_test = (y_test == 5)
y_train = (y_train == 5)
scaler = StandardScaler()
image_size = 784 # 28 x 28
X_train = X_train.reshape(X_train.shape[0], image_size)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
per_clf = Perceptron(random_state=42)
print(cross_val_score(per_clf, X_train_scaled, y_train, cv=5, scoring="accuracy"))
print(cross_val_score(per_clf, X_train_scaled, y_train, cv=5, scoring="precision"))
"""

###MLP###

#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

image_size = 784 # 28 x 28
#X_train = X_train.reshape(X_train.shape[0], image_size)



import tensorflow as tf
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z


from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    #hidden3 = fully_connected(hidden2, n_hidden3, scope="hidden3")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#learning_rate = 0.01


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
#print(mnist)

import tensorflow_datasets as tfds

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']
assert isinstance(train_data, tf.data.Dataset)
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000
# You can also access a builder directly
builder = tfds.builder("mnist")
assert builder.info.splits['train'].num_examples == 60000
builder.download_and_prepare()
datasets = builder.as_dataset()

# If you need NumPy arrays
np_datasets = tfds.as_numpy(datasets)
#for animal in np_datasets['train']:
    #print(animal.num)

mnist = np_datasets

from itertools import islice
#print(list(islice(mnist['train'], 7,10)))
#print(count)
#count = 0
#for x in mnist['train']:
#    count += 1
#print(count)




#results = [x for x in (itertools.islice(mnist['train'], 10))]
#list_1 = list(itertools.combinations(mnist['train'],2))
#print(list_1)

#for item in islice(mnist['train'], 7):
#    print(item)
#    print("hello"

#
#print(list(islice(mnist['train'], 7,10)))
#print(count)

train = list(mnist['train'])
test = list(mnist['test'])
def mlp_network(learning_rate, n_epochs, batch_size):
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(60000 // batch_size):
                #X_batch, y_batch = mnist['train'].next_batch(batch_size)
                list2 = list(islice(train, batch_size))
                X_batch = [item['image'] for item in list2]
                y_batch = [item['label'] for item in list2]
                new_batch = []
                for item in X_batch:
                    new_batch.append(item.reshape(784))
                sess.run(training_op, feed_dict={X: new_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: new_batch, y: y_batch})
            X_test = [item['image'] for item in test]
            y_test = [item['label'] for item in test]
            new_test = []
            for item in X_test:
                new_test.append(item.reshape(784))
            acc_test = accuracy.eval(feed_dict={X: new_test,
            y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./my_model_final.ckpt")


#feature_columns = tf.feature_column.numeric_column(X_train)
#feature_columns = tf.parse_example(X_train, features=tf.feature_column.make_parse_example_spec(columns))
#dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_columns)
#dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

#from sklearn.metrics import accuracy_score
#y_pred = list(dnn_clf.predict(X_test))
#accuracy_score(y_test, y_pred)


# If you need NumPy arrays
#np_datasets = tfds.as_numpy(datasets)
