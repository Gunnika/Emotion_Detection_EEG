from load_data import convertData
from convert_labels import convertLabels
from knn_classifier import KNNClassifier

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