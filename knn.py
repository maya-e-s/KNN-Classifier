# Maya Stevenson, 10/6/21, KNN Classifier
# The purpose of this assignment was to implement k-NN classification on a Letter Recognition Data Set 
# from the UCI Machine Learning Repository without using Python machine learning libraries like Scikit-learn. 
# This dataset contains 20,000 examples with the goal of identifying one of the 26 capital letters in the English alphabet.
# Link to dataset: https://archive.ics.uci.edu/ml/datasets/letter+recognition

import timeit, random, numpy as np 
import matplotlib.pyplot as plt, seaborn as sn
from sys import exit

# Preprocess the training data by centering (subtract mean) and scaling (divide by max)
# input: np.ndarray of training examples
# outputs: preprocessed np.ndarray of training examples, average and max of training examples for use in preprocessing test data
def preprocess_train(train_x):
    avg = np.sum(train_x, axis=0)/train_x.shape[0]
    max_val = np.amax(train_x, axis=0)
    train_x = (train_x-avg)/max_val
    return train_x, avg, max_val

# Preprocess the test data using the mean and max found in the training data
# inputs: np.ndarray of testing examples, average and max of training examples
# outputs: preprocessed np.ndarray of test examples
def preprocess_test(test_x, avg, max_val):
    test_x = test_x - avg
    test_x = test_x/max_val
    return test_x

# Return index of letter in alphabet (ex: A --> 1); used for reading in data file 
def letter_to_int(letter):
    return ord(letter)-64

# KNN Classifier: Compute the k nearest neighbors and assign class by majority vote 
# inputs: np.ndarray of training examples (train_x) with labels (train_y), np.ndarray of testing examples (test_x), k parameter (num_nn)
# output: prediction of labels for testing examples
def knn_classify(train_x, train_y, test_x, num_nn):
    pred_y = np.zeros([test_x.shape[0], ])
    # for each row (example) in test data, calculate distance between test point and training points
    for i in range(test_x.shape[0]): 
        dist = np.sqrt(np.sum((test_x[i,:] - train_x)**2, axis=1))
        dist_label = np.vstack([dist, train_y]) # append labels of training data
        # sort neighbors by smallest distance and keep k nearest neighbors; assign class by majority vote 
        neighbors = dist_label[:, dist_label[0, :].argsort()] 
        nn = neighbors[:, :num_nn]
        pred_y[i] = majority(nn[1,:], nn[0,:])
    return pred_y

# Helper function for KNN, uses majority vote to decide predicted label 
# Note: in case of tie, choose label with majority vote and smallest distance
# inputs: np.ndarray of labels and associated distances
# output: label of the majority vote for one test point
def majority(label, dist):
    u_label, count = np.unique(label, return_counts=True)
    winner = np.argwhere(count==np.amax(count)) # indices of max count
    # if there is a tie, choose label with majority vote and smallest distance
    if len(winner) > 1:
        dist_sum = list()
        for cand in u_label[winner]: # for each candidate with a majority vote
            dist_sum.append(np.sum(dist[np.argwhere(label==cand)])) # get and sum distances
        return u_label[np.argmin(dist_sum)] # return label with minimum distance
    # else majority rules
    else:
        return u_label[winner]

# Compute accuracy of the KNN Classifier 
# inputs: test labels (test_y) and predicted labels (pred_y)
# output: the classification accuracy as a float between 0.0 and 1.0
def compute_accuracy(test_y, pred_y):
    count = np.sum(test_y==pred_y)
    acc = count/test_y.shape[0] 
    return acc

# Create and display a confusion matrix
# inputs: test labels (test_y) and predicted labels (pred_y)
# output: np.ndarray containing confusion matrix values without labels
def confusion_matrix(test_y, pred_y):
    labels = [chr(x) for x in range(65, 91)] # 'A' to 'Z'
    mat = np.zeros([len(labels), len(labels)])
    # fill confusion matrix
    for i in range(test_y.shape[0]):
        row = int(test_y[i]) - 1
        col = int(pred_y[i]) - 1 
        mat[row, col] = mat[row, col] + 1
    # create and display plot
    sn.heatmap(mat, annot=True, fmt='g', annot_kws={"size": 8}, xticklabels=labels, yticklabels=labels, cmap='Blues', cbar_kws={'label': 'Predictions'})
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.title('Confusion Matrix for KNN Classifier', fontsize=20)
    plt.show()
    return mat

# User Menu called at start of program; allows user to select KNN parameters and prints results 
def user_menu():   
    print('\n=================================== Menu ==================================\nWelcome to the KNN Classifier for multiclass letter recogition.')
    # Get user input to select data file, read data file into a np.ndarray and convert letter labels into integers  
    fname = input("Enter path of 'letter-recognition.data', or click 'enter' if data is in current folder: ") or "letter-recognition.data"
    try:
        input_data = np.loadtxt(fname, delimiter=',', converters={0: letter_to_int})
    except:
        print('File "', fname, '" cannot be opened. Quitting program.', sep='')
        quit()
    # Split data into training and testing 
    train_x = input_data[0:15000, 1:] # examples set aside for training
    train_y = input_data[0:15000, 0] # label vector for training examples
    test_x = input_data[-5000:, 1:] # examples set aside for testing
    test_y = input_data[-5000:, 0] # label vector for testing examples
    # User chooses number of training samples to use (out of the 15000 set aside for training) 
    while True:
        try:
            num_train = int(input("Choose number of training samples (between 5000 and 15000, click 'enter' to default to 10000): ") or 10000) # training samples
            if num_train<5000 or num_train>15000: raise ValueError
            break
        except ValueError: # wrong data type or out of range
            print("Sorry, that input is invalid. Try again.")
            continue
    # User chooses number of nearest neighbors (k value, or num_nn) 
    while True: 
        try:
            num_nn = int(input("Choose number of nearest neighbors (between 1 and 5, click 'enter' to default to 3): ") or 3) # nearest neighbors
            if num_nn<1 or num_nn>5: raise ValueError
            break
        except ValueError: # wrong data type or out of range
            print("Sorry, that input is invalid. Try again.")
            continue

    # Run KNN Classifier
    start_time = timeit.default_timer()
    # Get num_train random samples as a subset from training set
    idx =  random.sample(range(0, 15000), num_train)
    train_x = train_x[idx] 
    train_y = train_y[idx] # matching labels
    # Preprocess training and testing data
    train_x, avg, max_val = preprocess_train(train_x)
    test_x = preprocess_test(test_x, avg, max_val)
    # Make predictions on test data with KNN Classifier and print accuracy 
    print('\n============================== KNN Classifier ==============================')
    print("Testing KNN Classifier with",num_nn, "nearest neighbor(s) and", num_train, "training samples...")
    knn_pred = knn_classify(train_x, train_y, test_x, num_nn)
    knn_acc = compute_accuracy(test_y, knn_pred)
    stop_time = timeit.default_timer() 
    print("This KNN Classifier took", stop_time - start_time, "seconds to run and has an accuracy of", knn_acc)
    # Ask user if they would like to see confusion matrix or end program
    while True: 
        try:
            conf = int(input('Would you like to see a confusion matrix? Enter 1 for yes, and 0 to end program: '))
            if conf<0 or conf>1: raise ValueError
            break
        except ValueError: # wrong data type or out of range
            print("Sorry, that input is invalid. Try again.")
            continue
    if conf: confusion_matrix(test_y, knn_pred)
    # end program 
    print('Goodbye.')
    return None

if __name__ == "__main__":    
    user_menu()