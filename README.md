<h1>Iris Flower Pattern Recognition using K-Nearest Neighbors (KNN)</h1>

<p>
This project is a pattern recognition project implemented using machine learning techniques. It focuses on classifying Iris flowers into different species based on their features. The dataset used in this project is the Iris dataset, which can be downloaded from Kaggle at <a href='https://www.kaggle.com/datasets/dev0914sharma/customer-clustering'>this link.<a/>

</p>
<h2>Project Overview</h2>
<p>
  
The project is developed in Python programming language and consists of two files:

1. main.py: This is the main source code file that contains the implementation of the pattern recognition algorithm using KNN.<br>
2. IRIS.csv: This file is the dataset file, which contains the data of Iris flowers.
  
  </p>
<h2>Project Workflow</h2>
<p>
  
The project follows the following workflow:

1. Loading the dataset: The load_data function reads the dataset from the IRIS.csv file and prepares it for further processing. It converts the labels of the       flower types from strings to float numbers for easier handling.<br>
2. Splitting the data: The split_data function separates the features and labels into two lists, X and y, respectively.<br>
3. Train-test split: The split_train_test function randomly selects a portion of the data for training and keeps the rest for testing. It returns four lists: train_X, train_y, test_X, and test_y.<br>
4. Distance calculation: The euclidean_distance function calculates the Euclidean distance between two objects using their features.<br>
5. KNN classifier: The KNN class represents a KNN classifier with k = 3. It contains a fit function to train the classifier on the training data and a predict function to predict the labels for the test data.<br>
6. Model training and evaluation: The main part of the code loads the dataset, performs the train-test split, initializes the KNN classifier with k = 3, fits the classifier on the training data, and predicts the labels for the test data. It then calculates the accuracy of the model by comparing the predicted labels with the actual labels.<br>
7. Sample testing: The code includes an additional section to test the classifier on sample data points and print the predicted class name and value.

</p>  
<h2>Dependencies</h2>
<p>
  
The project uses the following packages and libraries:

. csv: Used to read and process the dataset.<br>
. random: Used for random data sampling during the train-test split.<br>
. math: Used for mathematical calculations.<br>
. pandas: Used for printing and visualizing the dataset.<br>
  
</p>
<h2>Usage</h2>
<p>
  
To use this project, follow these steps:

1. Download the dataset file IRIS.csv from the provided link and place it in the same directory as the main.py file.<br>
2. Run the main.py file using Python.<br>
3. The program will load the dataset, perform the train-test split, train the KNN classifier, predict the labels for the test data, and calculate the accuracy of the model.<br>
4. Additionally, the program will test the classifier on some sample data points and display the predicted class name and value.
  
</p>
<p>
Please note that you may need to install the required dependencies if they are not already installed in your Python environment.<br>

That's it! You now have a pattern recognition project for classifying Iris flowers using KNN.<br>
<p>
<hr>
