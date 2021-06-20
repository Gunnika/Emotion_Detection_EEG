from load_data import convertData
from convert_labels import convertLabels
from knn_classifier import run_knn_classifier
from decision_tree import run_decision_tree
from svm_pca import run_svm_pca_classifier

#ignore warnings 
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("Starting Program \n")
    convertData()
    print("Encoding classes \n")
    convertLabels()
    print("\n KNN Classification")
    run_knn_classifier()
    print("\n Decision Tree Classification")
    run_decision_tree()
    print("\n SVM with PCA Classification")
    run_svm_pca_classifier()
