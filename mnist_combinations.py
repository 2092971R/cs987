'''
    File name: mnist.py
    Author: Team Pi
    Date created: 04/04/2019
    Assignment: Assignment 2, Machine Learning for Data Analytics

    Networks for the MNIST dataset
'''


#Convolutional Networks
def cnn(combination, learning_rate, n_epochs, batches):
    import tensorflow_datasets as tfds
    import numpy as np
    from tensorflow.keras import datasets, layers, models, callbacks
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint

    #import the data
    builder = tfds.builder("mnist")
    assert builder.info.splits['train'].num_examples == 60000
    builder.download_and_prepare()
    datasets = builder.as_dataset()
    np_datasets = tfds.as_numpy(datasets)
    mnist = np_datasets

    #get the test and training datset splits
    train = list(mnist['train'])
    test = list(mnist['test'])
    X_train = [item['image'] for item in train]
    X_train = np.asarray(X_train, dtype=np.float32)
    train_images = X_train / 255.0

    y_train = [item['label'] for item in train]
    train_labels = np.asarray(y_train, dtype=np.int32)

    X_test = [item['image'] for item in test]
    X_test = np.asarray(X_test, dtype=np.float32)
    test_images = X_test / 255.0

    y_test = [item['label'] for item in test]
    test_labels = np.asarray(y_test, dtype=np.int32)

    #create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    if (combination == 2):
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if (combination == 4):
        model.add(layers.Dropout(0.9))
    else:
        model.add(layers.Dropout(0.4))
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()

    #created ckpt file
    if (combination == 1):
        checkpoint_path = "mnist-1-"+ str(learning_rate) + "-" +str(n_epochs) + "-" + str(batches) + ".ckpt"
        #checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = ModelCheckpoint(checkpoint_path,save_weights_only=True,
                                                         verbose=1)
        #tbCallBack = callbacks.TensorBoard(log_dir=tb, histogram_freq=0, write_graph=True, write_images=True)
        model.fit(train_images, train_labels, epochs=n_epochs, verbose=0, batch_size=60000 // batches,
                  callbacks=[cp_callback])
    else:
        model.fit(train_images, train_labels, epochs=n_epochs, verbose = 0, batch_size = 60000//batches, callbacks=[tbCallBack])
    train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0)
    print("Training Accuracy:",train_acc)
    print("Testing Accuracy:", test_acc)


def mlp(combination, learning_rate, n_epochs, batches):
    from itertools import islice
    import tensorflow_datasets as tfds
    from tensorflow.contrib.layers import fully_connected
    import numpy as np
    from tensorflow.nn import sigmoid, relu, sparse_softmax_cross_entropy_with_logits, in_top_k
    from tensorflow import placeholder,float32, int64, name_scope, reduce_mean, global_variables_initializer, cast, Session
    from tensorflow.train import Saver, GradientDescentOptimizer

    #import the data
    builder = tfds.builder("mnist")
    assert builder.info.splits['train'].num_examples == 60000
    builder.download_and_prepare()
    datasets = builder.as_dataset()
    np_datasets = tfds.as_numpy(datasets)
    mnist = np_datasets
    train = list(mnist['train'])
    test = list(mnist['test'])

    #define the placeholders

    X = placeholder(float32, shape=(None, 784), name="X")
    y = placeholder(int64, shape=(None), name="y")

    with name_scope("mlp"):
        hidden1 = fully_connected(X, 100, scope="hidden1", activation_fn=sigmoid)
        if combination == 2:
            hidden2 = fully_connected(hidden1, 100, scope="hidden2", activation_fn=sigmoid)
            hidden3 = fully_connected(hidden2, 100, scope="hidden3", activation_fn=sigmoid)
            logits = fully_connected(hidden3, 10, scope="outputs", activation_fn=None)
        else:
            logits = fully_connected(hidden1, 10, scope="outputs", activation_fn=None)

    with name_scope("loss"):
        xentropy = sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = reduce_mean(xentropy, name="loss")

    with name_scope("eval"):
        correct = in_top_k(logits, y, 1)
        accuracy = reduce_mean(cast(correct, float32))

    init = global_variables_initializer()
    saver = Saver()

    with name_scope("train"):
        optimizer = GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    with Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(batches):
                temp = list(islice(train, 60000 // batches))
                X_train = [item['image'] for item in temp]
                y_train = [item['label'] for item in temp]
                new_batch = []
                for item in X_train:
                    new_batch.append(item.reshape(784))
                sess.run(training_op, feed_dict={X: new_batch, y: y_train})
            acc_train = accuracy.eval(feed_dict={X: new_batch, y: y_train})
            X_test = [item['image'] for item in test]
            y_test = [item['label'] for item in test]
            new_test = []
            for item in X_test:
                new_test.append(item.reshape(784))
            acc_test = accuracy.eval(feed_dict={X: new_test, y: y_test})
        print("Training accuracy:", acc_train)
        print("Testing accuracy:", acc_test)
        #save_path = saver.save(sess, "./my_model_final.ckpt")

