from load_data import convertData
from convert_labels import convertLabels
from knn_classifier import KNNClassifier
from decision_tree import decision_tree
from svm_pca import svm_pca_classifier

#ignore warnings 
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("Starting Program \n")
    convertData()
    print("Encoding classes \n")
    convertLabels()
    print("KNN Classification")
    KNNClassifier()
    print("Decision Tree Classification")
    KNNClassifier()
    print("SVM with PCA Classification")
    KNNClassifier()
