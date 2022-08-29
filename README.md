# KNN Classifier
The purpose of this program is to implement k-NN classification on a [Letter Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/letter+recognition) from the UCI Machine Learning Repository without using Python machine learning libraries like Scikit-learn. This dataset contains 20,000 examples, with 15,000 set aside for training and the remaining 5,000 set aside for testing. The goal of the k-NN Classifier is to identifying one of the 26 capital letters in the English alphabet.

## K-NN Classification
K-NN works by computing the distances between a test point and all the training examples and assigning the test point a label based on the majority vote of the k nearest neighbors. In case of a tie, the label with a majority vote and the smallest summed distance was chosen. 

## User Menu
After starting the program, a user menu is printed to allow the user to enter the file path of the data along with picking a subset of the 15,000 training examples and a k value. After classifying the data, the runtime and accuracy of the classification is printed, and the user is asked if they would like to see a confusion matrix. 

## Results
The biggest effect on accuracy for k-NN was the number of training examples. As more training samples were used, the accuracy increased. Choosing k between 1 and 3 yeilded the highest accuracy, but changing k was not nearly as significant as changing the number of training examples. 

## Example Output
### K-NN with 15,000 training examples and k=3
![terminal output 3NN with 15000 training points](https://user-images.githubusercontent.com/76231198/187271452-0bd29786-1f38-41e0-9c45-9cfcd6233597.png)
![confusion matrix for 3NN with 15000 training points](https://user-images.githubusercontent.com/76231198/187271488-e21d879e-627c-473a-88e5-b22620cb3c1e.png)

### K-NN with 7,000 training examples and k=4
![terminal output 4NN with 7000 training points](https://user-images.githubusercontent.com/76231198/187271509-0591f72a-fa32-4bb6-af6f-f56a181793dd.png)
