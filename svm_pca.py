# import libraries for SVM
import numpy as np
from sklearn.model_selection import train_test_split ###
from sklearn.svm import SVC  
from sklearn.decomposition import PCA
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

#build SVM Classifier with PCA dimensionality reduction
def svm_pca_classifier(data,label): 
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    
    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # dimensionality reduction with PCA
    pca = PCA(n_components=50)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # SVM classifier with kernel = rbf
    classifier = SVC(kernel='rbf', gamma=0.01, C=100)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred)) ### evaluate performance
    print(accuracy_score(y_test, y_pred)*100) ### evaluate performance

def run_svm_pca_classifier():
    # call the function svm_pca_classifier()
    print("VALENCE \n")
    svm_pca_classifier(x_vals, y_vals_0)
    print("AROUSAL \n")
    svm_pca_classifier(x_vals, y_vals_1)
    print("DOMINANCE \n")
    svm_pca_classifier(x_vals, y_vals_2)
    print("LIKING \n")
    svm_pca_classifier(x_vals, y_vals_3)
