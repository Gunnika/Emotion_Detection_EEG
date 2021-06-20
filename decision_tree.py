# import libraries for decision tree
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

x = 'data/features.dat'
y_0 = 'data/labels_class_0.dat'
y_1 = 'data/labels_class_1.dat'
y_2 = 'data/labels_class_2.dat'
y_3 = 'data/labels_class_3.dat'

#load data from text file in numpy array
x_vals = np.loadtxt(x)
y_vals_0 = np.loadtxt(y_0)
y_vals_1 = np.loadtxt(y_1)
y_vals_2 = np.loadtxt(y_2)
y_vals_3 = np.loadtxt(y_3)

def decision_tree(data, label): 
    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)
    #create scaler
    scaler = StandardScaler()
    #fit scaler to training data
    scaler.fit(X_train)
    #transform training and testing data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #create classifier
    clf = tree.DecisionTreeClassifier()
    #train classifier
    clf.fit(X_train, y_train)
    #predict labels of test data
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    # call the function knn_classifier()
    print("VALENCE \n")
    decision_tree(x_vals, y_vals_0)
    print("AROUSAL \n")
    decision_tree(x_vals, y_vals_1)
    print("DOMINANCE \n")
    decision_tree(x_vals, y_vals_2)
    print("LIKING \n")
    decision_tree(x_vals, y_vals_3)