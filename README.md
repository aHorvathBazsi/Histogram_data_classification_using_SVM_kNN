# Histogram_data_classification_using_SVM_kNN
The attached script presents a simple method to classify histogram data. The data was extracted from videos taken about 3 different people (the algorithm actually is a multi-class classification algorithm on 3 classes)
HistogramData.mat was generated using Matlab from 3 videos (each video contained a person walking towards the camera)
The main goal of the problem is to detect each person based on some extracted features (in this example histogram is used, but you could also use features like SURF, MSER or HarrisFeatures etc.
HistogramData.mat stores data in a 3-dimensional matrices in the following way: person:observation:features (first channel is to identify which person is on the video (in the problem used for labels), second is the number of observation (an observation is actually a frame from the original video) and the last channel is for the actual feature vector.
HistogramData.mat contains different sets of data, for instance HistogramData['hist_train_30'] contain the training set of vector features (30 indicates the length of the vector feature, in this case 30, which refers to the fact that the original frame was kept online 30 color (subquantization) )
Unfortunately HistogramData.mat is private, but if you can generate a similar matrix (is you keep the structure of the data) you could easily use the source code to implement multiclass classification using k-Nearest Neighbours or Support Vector Machine.

Please note that the comments for the code is in romanian, it will be updated soon.
