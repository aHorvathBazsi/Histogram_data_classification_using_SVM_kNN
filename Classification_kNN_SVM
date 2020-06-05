#importarea librăriilor folosite
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
%matplotlib inline

#incărcare fișier .mat
data = scipy.io.loadmat('HistogramData.mat')

#extragere număr observații
observations_train_30 = data['hist_train_30'].shape[1]
observations_test_30 = data['hist_test_30'].shape[1]
observations_train_128 = data['hist_train_128'].shape[1]
observations_test_128 = data['hist_test_128'].shape[1]
observations_train_255 = data['hist_train_255'].shape[1]
observations_test_255 = data['hist_test_255'].shape[1]

#extragere X_train și X_test, reshape pentru a reduce prima dimensiune
X_train_30 = data['hist_train_30'].reshape(-1,30)
X_test_30 = data['hist_test_30'].reshape(-1,30)
X_train_128 = data['hist_train_128'].reshape(-1,128)
X_test_128 = data['hist_test_128'].reshape(-1,128)
X_train_255 = data['hist_train_255'].reshape(-1,255)
X_test_255 = data['hist_test_255'].reshape(-1,255)

#Generarea etichetelor
Y_train_30 = np.zeros((X_train_30.shape[0],1),dtype = np.uint8)
Y_test_30 = np.zeros((X_test_30.shape[0],1),dtype = np.uint8)
Y_train_128 = np.zeros((X_train_128.shape[0],1),dtype = np.uint8)
Y_test_128 = np.zeros((X_test_128.shape[0],1),dtype = np.uint8)
Y_train_255 = np.zeros((X_train_255.shape[0],1),dtype = np.uint8)
Y_test_255 = np.zeros((X_test_255.shape[0],1),dtype = np.uint8)

Y_train_30[observations_train_30:2*observations_train_30]=1
Y_train_30[2*observations_train_30:3*observations_train_30]=2
Y_train_128[observations_train_128:2*observations_train_128]=1
Y_train_128[2*observations_train_128:3*observations_train_128]=2
Y_train_255[observations_train_255:2*observations_train_255]=1
Y_train_255[2*observations_train_255:3*observations_train_255]=2

Y_test_30[observations_test_30:2*observations_test_30]=1
Y_test_30[2*observations_test_30:3*observations_test_30]=2
Y_test_128[observations_test_128:2*observations_test_128]=1
Y_test_128[2*observations_test_128:3*observations_test_128]=2
Y_test_255[observations_test_255:2*observations_test_255]=1
Y_test_255[2*observations_test_255:3*observations_test_255]=2

#rescalare pentru features 
scaler=StandardScaler()
scaler.fit(X_train_30)
scaler.transform(X_train_30)
scaler.transform(X_test_30)
scaler=StandardScaler()
scaler.fit(X_train_128)
scaler.transform(X_train_128)
scaler.transform(X_test_128)
scaler=StandardScaler()
scaler.fit(X_train_255)
scaler.transform(X_train_255)
scaler.transform(X_test_255)

# definirea modelului de LinearSVC
from sklearn.svm import LinearSVC,SVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier
model = LinearSVC()
# antrenare pe setul de date hist_train_30
model.fit(X_train_30,Y_train_30)
predictions = model.predict(X_test_30)
# afișarea rezultatelor pentru hist_train_30
print(confusion_matrix(Y_test_30,predictions))
print(classification_report(Y_test_30,predictions))
# antrenare pe setul de date hist_train_30
model.fit(X_train_128,Y_train_128)
predictions = model.predict(X_test_128)
# afișarea rezultatelor pentru hist_train_128
print(confusion_matrix(Y_test_128,predictions))
print(classification_report(Y_test_128,predictions))
# antrenare pe setul de date hist_train_255
model.fit(X_train_255,Y_train_255)
predictions = model.predict(X_test_255)
# afișarea rezultatelor pentru hist_train_255
print(confusion_matrix(Y_test_255,predictions))
print(classification_report(Y_test_255,predictions))

# Clasificare folosind KNN
model = KNeighborsClassifier(n_neighbors=3)
# antrenare pe setul de date hist_train_30
model.fit(X_train_30,Y_train_30.reshape(-1,))
predictions = model.predict(X_test_30)
# afișarea rezultatelor pentru hist_train_30
print(confusion_matrix(Y_test_30,predictions))
print(classification_report(Y_test_30,predictions))
# antrenare pe setul de date hist_train_30
model.fit(X_train_128,Y_train_128.reshape(-1,))
predictions = model.predict(X_test_128)
# afișarea rezultatelor pentru hist_train_128
print(confusion_matrix(Y_test_128,predictions))
print(classification_report(Y_test_128,predictions))
# antrenare pe setul de date hist_train_255
model.fit(X_train_255,Y_train_255.reshape(-1,))
predictions = model.predict(X_test_255)
# afișarea rezultatelor pentru hist_train_255
print(confusion_matrix(Y_test_255,predictions))
print(classification_report(Y_test_255,predictions))
