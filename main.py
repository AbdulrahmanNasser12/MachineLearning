import csv
import random
import math
import pandas as pd


#------------------------------------
# To View the dataset
   # data = pd.read_csv('./IRIS.csv')
   # print(data)
#------------------------------------
#loading dataset
def load_data(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)[1:]
        #changing lables from strings into float
        for i in range(len(dataset)):
            if dataset[i][-1] == 'Iris-setosa':
                dataset[i][-1] = 0
            elif dataset[i][-1] == 'Iris-versicolor':
                dataset[i][-1] = 1
            elif dataset[i][-1] == 'Iris-virginica':
                dataset[i][-1] = 2
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#separating features and labels into two lists 
def split_data(data):
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i][:-1])
        y.append(data[i][-1])
    return X, y

#take some values as trainging randomly
def split_train_test(X, y, split_ratio):
    train_size = int(len(X) * split_ratio)
    train_X = []
    train_y = []
    test_X = X.copy()
    test_y = y.copy()
    while len(train_X) < train_size:
        index = random.randrange(len(test_X))
        train_X.append(test_X.pop(index))
        train_y.append(test_y.pop(index))
    return train_X, train_y, test_X, test_y

#Calculating destace between two objects
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)

#KNN classifier class with k = 3
class KNN:
    def __init__(self, k=3):
        self.k = k
    #fiting k lists with train lists
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    #predicting the true label using euclidean_distance function
    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                distance = euclidean_distance(X_test[i], self.X_train[j])
                distances.append((distance, self.y_train[j]))
            distances.sort()
            neighbors = distances[:self.k]
            classes = {}
            for neighbor in neighbors:
                if neighbor[1] in classes:
                    classes[neighbor[1]] += 1
                else:
                    classes[neighbor[1]] = 1
            prediction = max(classes, key=classes.get)
            predictions.append(int(prediction))
        return predictions

# Main
data = load_data('./IRIS.csv')
X, y = split_data(data)
train_X, train_y, test_X, test_y = split_train_test(X, y, 0.9)
knn = KNN(k=3)
knn.fit(train_X, train_y)
predictions = knn.predict(test_X)
accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == test_y[i]) / float(len(predictions))

# testing code using samples
sample1 = [5.1, 3.5, 1.4, 0.2]
sample2 = [6.7, 3.0, 5.2, 2.3]
sample3 = [6.4, 3.2, 4.5, 1.5]
sample = [sample1, sample2, sample3]

for i in sample:
    prediction = knn.predict([i])
    iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predicted_iris_name = iris_names[prediction[0]]
    print('Predicted class name:', predicted_iris_name)
    print('Predicted class value:', prediction)

print('Classifier accuracy:', accuracy)
