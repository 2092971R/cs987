'''
    File name: mnist.py
    Author: Team Pi
    Date created: 04/04/2019
    Assignment: Assignment 2, Machine Learning for Data Analytics
'''

import argparse
from mnist_combinations import mlp, cnn

def network_one(learning_rate, epochs, batches):

    print("CNN with 1 convolutional layer and 2 fully connected layers")
    print("Combination One with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    cnn(1, learning_rate, epochs, batches)


def network_two(learning_rate, epochs, batches):

    print("Multi layer perceptron with 3 hidden layers")
    print("Combination Two with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    mlp(1, learning_rate, epochs, batches)

def network_three(learning_rate, epochs, batches):

    print("CNN with 2 convolutional layers and 1 fully connected layer")
    print("Combination Three with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    cnn(2,learning_rate, epochs, batches)

def network_four(learning_rate, epochs, batches):

    print("Multilayer perceptron with 3 hidden layers")
    print("Combination Four with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    mlp(2,learning_rate, epochs, batches)

def network_five(learning_rate, epochs, batches):

    print("CNN with 1 convolutional layer and 1 fully connected layer")
    print("Combination Five with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    cnn(3,learning_rate, epochs, batches)

def network_six(learning_rate, epochs, batches):

    print("CNN with 1 convolutional layer and 1 fully connected layer")
    print("Combination Six with learning rate: {} epochs: {} and batches: {}".format(learning_rate, epochs, batches))
    cnn(4,learning_rate, epochs, batches)


def main(combination, learning_rate, epochs, batches, seeds):
    # Set Seed
    print("Seed: {}".format(seeds))
    from numpy.random import seed
    from tensorflow import set_random_seed
    seed(seeds)
    set_random_seed(seeds)

    if int(combination)==1:
        network_one(learning_rate, epochs, batches)

    if int(combination)==2:
        network_two(learning_rate, epochs, batches)

    if int(combination)==3:
        network_three(learning_rate, epochs, batches)

    if int(combination)==4:
        network_four(learning_rate, epochs, batches)

    if int(combination)==5:
        network_five(learning_rate, epochs, batches)

    if int(combination)==6:
        network_six(learning_rate, epochs, batches)

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, int(epochs), int(batches), int(seed))


