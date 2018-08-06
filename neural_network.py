import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# True positives, false positives, false negatives

def calc_true_pos(pred, y_test_data):
    true_positive = []
    for i in range(0, len(y_test_data)): # Iterate through all of our predictions
        if (pred[i] == 1 and y_test_data[i] == pred[i]): # If we predict malignant and our prediction is correct
            true_positive.append(pred[i]) # then this prediction is a true positive
    num_true_pos = len(true_positive)
    return num_true_pos
            
def calc_false_pos(pred, y_test_data):
    false_positive = []
    for i in range(0, len(y_test_data)): # Iterate through all of our predictions
        if (pred[i] == 1 and y_test_data[i] != pred[i]): # If we predict malignant and our prediction is incorrect
            false_positive.append(y_test_data[i]) # then this prediction is a false positive
    num_false_pos = len(false_positive)
    return num_false_pos

def calc_false_neg(pred, y_test_data):
    false_negative = []
    for i in range(0, len(y_test_data)):
        if (pred[i] == 0 and y_test_data[i] != pred[i]):
            false_negative.append(y_test_data[i])
    num_false_neg = len(false_negative)
    return num_false_neg

# Accuracy, precision, recall, F1 score

def calculate_accuracy(pred, y_test_data):
    return accuracy_score(pred, y_test_data)

def calculate_precision(pred, y_test_data):
    num_true_pos = calc_true_pos(pred, y_test_data)
    num_false_pos = calc_false_pos(pred, y_test_data)
    precision = (num_true_pos)/(num_true_pos + num_false_pos)
    return precision

def calculate_recall(pred, y_test_data):
    num_true_pos = calc_true_pos(pred, y_test_data)
    num_false_neg = calc_false_neg(pred, y_test_data)
    recall = (num_true_pos)/(num_true_pos + num_false_neg)
    return recall

def calculate_f1(pred, y_test_data):
    precision = calculate_precision(pred, y_test_data)
    recall = calculate_recall(pred, y_test_data)
    f1 = 2 * (precision * recall)/(precision + recall)
    return f1
    
   mnist = datasets.load_digits() # Load the MNIST dataset

X = mnist.data # The input value is the pixelated data
y = mnist.target # The y-values are the correct labels

X_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.20)

neural_network = MLPClassifier(activation='relu', max_iter = 500, learning_rate_init=0.001)

neural_network.fit(X_train, y_train)

y_pred = neural_network.predict(x_test)

accuracy = calculate_accuracy(y_pred, y_test)
print("Accuracy: ")
print(accuracy)

#^the only one that matters in this case because... something Lainey said

precision = calculate_precision(y_pred, y_test)
print("Precision: ")
print(precision)

recall = calculate_recall(y_pred, y_test)
print("Recall: ")
print(recall)

f1 = calculate_f1(y_pred, y_test)
print("F1: ")
print(f1)
