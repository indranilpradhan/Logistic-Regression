import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys

def apply_pca(train):
    n_components = 74
    x_mean = np.mean(train, axis =0)
    x_center = train - x_mean
    x_cov = np.cov(x_center.T)
    x_eigenvalues, x_eigenvectors = np.linalg.eig(x_cov)
    indexes = x_eigenvalues.argsort()[::-1]   
    eigenvalues = x_eigenvalues[indexes]
    eigenvectors = x_eigenvectors[:,indexes]
    red_eigenvec = eigenvectors[:,:n_components]
    x_pca = red_eigenvec.T.dot(train.T)
    global_eigenvector = eigenvectors
    return x_pca.T,eigenvectors

def transform_pca(eigenvectors, test):
    n_components = 74
    red_eigenvec = eigenvectors[:,:n_components]
    test_pca = red_eigenvec.T.dot(test.T)
    return test_pca.T

def accuracy(predictions, y):
    return ((predictions == y).mean()*100)

def sigmoid_function(x):
    g = 1/(1 + np.exp(-x))
    return g

def calculate_cost(x,y,w,h):
    total_cost = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    cost = total_cost/len(y)
    return cost

def claculate_gradient(x,y,h):
    gradient = (y-h).dot(x)/len(y)
    return gradient

def train_data(xtrain,y,numofiter,learningrate):
    x = np.ones(shape=(xtrain.shape[0], xtrain.shape[1] + 1),dtype=complex)
    x[:, 1:] = xtrain
    classes = np.unique(y)
    weights =[]
    for cls in classes:
        weight = np.zeros(x.shape[1],dtype=complex)
        y_map = []
        for i in y:
            if(i==cls):
                y_map.append(1)
            else:
                y_map.append(0)
        #print("ymap ",y_map)
        for i in range(numofiter):
            h = sigmoid_function(x.dot(weight))
            gradient = claculate_gradient(x,y_map,h)
            weight = weight + learningrate*gradient
        weights.append(weight)
    return weights

def predict_class(xval,y,weights):
    x = np.ones(shape=(xval.shape[0], xval.shape[1] + 1),dtype=complex)
    x[:, 1:] = xval
    temp_predictions = []
    for i in x:
        hypothesis = np.zeros(shape=(len(weights)),dtype=complex)
        k =0
        for weight in weights:
            h = sigmoid_function(i.dot(weight))
            #print("h===== ",h)
            hypothesis[k] = h
            k = k+1
        temp_predictions.append(np.argmax(hypothesis))
    predictions = []
    #print("t_pred ",temp_predictions)
    for indx in temp_predictions:
        #print("y_indx",indx)
        predictions.append(y[indx])
    return predictions

def label_to_ordinal_convert(train_labels):
    label_to_ordinal = {}
    uniq_label = np.unique(train_labels)
    for i in range(uniq_label.shape[0]):
        label_to_ordinal[uniq_label[i]] = i
    return label_to_ordinal

def ordinal_to_label_convert(train_labels):
    ordinal_to_label = {}
    uniq_label = np.unique(train_labels)
    for j in range(uniq_label.shape[0]):
        ordinal_to_label[j] = uniq_label[j]
    return ordinal_to_label

def train_labels_convert(train_labels, label_to_ordinal):
    train_ytrain = np.zeros(len(train_labels))
    for i in range(len(train_labels)):
        train_ytrain[i] = label_to_ordinal[train_labels[i]]
    return train_ytrain

def main():
    train_path = str(sys.argv[1])
    train_images = []
    train_labels = []
    train_file = open(train_path,"r")
    for train_line in train_file:
        path_label = train_line.split(" ")
        train_f = path_label[0]
        train_label = path_label[1].replace('\n', '')
        train_images.append(np.asarray(Image.open(train_f).convert('L').resize((64, 64))).flatten())
        train_labels.append(train_label)
    train_train = np.array(train_images)
    train_scalar = MinMaxScaler()
    train_train = train_scalar.fit_transform(train_train)
    train_train,train_eigenvects = apply_pca(train_train)
    train_labels = np.array(train_labels)
    train_file.close()

    train_xtrain = train_train[:,:]

    label_to_ordinal = label_to_ordinal_convert(train_labels)
    ordinal_to_label = ordinal_to_label_convert(train_labels)
    train_ytrain = train_labels_convert(train_labels, label_to_ordinal)
    test_path = str(sys.argv[2])
    test_images = []
    test_file = open(test_path,"r")
    for test_line in test_file:
        test_line= test_line.replace('\n','')
        test_images.append(np.asarray(Image.open(test_line).convert('L').resize( (64, 64))).flatten())
    test_test = np.array(test_images)
    test_test = train_scalar.transform(test_test)
    test_test = transform_pca(train_eigenvects,test_test)
    test_xtest = test_test[:,:]
    test_file.close()

    numofiter = 100000
    learningrate = 0.01
    t_weights = train_data(train_xtrain, train_ytrain, numofiter,learningrate)
    t_classes = np.unique(train_ytrain)
    predictions5 = predict_class(test_xtest,t_classes,t_weights)
    
    predictiion_class = []
    print("Predicted classes")
    for i in predictions5:
        predictiion_class.append(ordinal_to_label[i])
        print(ordinal_to_label[i])
    #print("Predicted labels ",prediction_class)

if __name__ == "__main__":
	main()