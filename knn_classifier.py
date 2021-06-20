# import libraries for building knn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

# building KNN classifier for different classes(valence, arousal, dominance, liking) separately
def knn_classifier(data, label):
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)

        # feature scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # KNN
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred)*100)

if __name__ == '__main__':
    # call the function knn_classifier()
    print("VALENCE \n")
    knn_classifier(x_vals, y_vals_0)
    print("AROUSAL \n")
    knn_classifier(x_vals, y_vals_1)
    print("DOMINANCE \n")
    knn_classifier(x_vals, y_vals_2)
    print("LIKING \n")
    knn_classifier(x_vals, y_vals_3)


