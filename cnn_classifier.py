import numpy as np
from sklearn.model_selection import train_test_split

# import Tensorflow Keras libraries for building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

x = 'data/features.dat'
y_0 = 'data/labels_class_0.dat'
y_1 = 'data/labels_class_1.dat'
y_2 = 'data/labels_class_2.dat'
y_3 = 'data/labels_class_3.dat'

#load data from text file in numpy array
x_vals = np.genfromtxt(x, delimiter=' ')
y_vals_0 = np.genfromtxt(y_0, delimiter=' ')
y_vals_1 = np.loadtxt(y_1)
y_vals_2 = np.loadtxt(y_2)
y_vals_3 = np.loadtxt(y_3)
   
#data size = 25804800
def cnn_classifier_eeg(data, label):
    #split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(data.shape[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





if __name__ == '__main__':
    # call the function knn_classifier()
    print("VALENCE \n")
    cnn_classifier_eeg(x_vals, y_vals_0)
    print("AROUSAL \n")
    cnn_classifier_eeg(x_vals, y_vals_1)
    print("DOMINANCE \n")
    cnn_classifier_eeg(x_vals, y_vals_2)
    print("LIKING \n")
    cnn_classifier_eeg(x_vals, y_vals_3)