# Machine Learning Framework - Team 7

This repository contains all the code related to the machine learning framework developed by us. It supports creating a feed forward neural network of arbitrary shape, to then train it for a classification task. The package provides multiple visualization functionalities to plot accuracy, loss, precision, F1 score and a confusion matrix. It also provides convenient methods of saving and loading a network for future use.

## Quick-start
Check out the demo [Jupyter Notebook](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/ml-package/-/blob/main/network_demo.ipynb) that is available for a quick demo.

## Set-up
1. Clone the project on your local machine
2. Create a new **!conda!** environment with python version 3.7
3. Activate the environment, navigate to the main directory of the project and install the requirements via
 `pip install -r requirements.txt`
4. Tensorflow must be installed in the local environment. Therefor use `pip install tensorflow`, since it did not work for us to include it in the `requirements.txt`.

## Usage

### Creating a network
A model can be created using the constructor of the Network class.  

    from network import Network
    
    network = Network(layer_shape=[784, 64, 64, 10])

  This will create a network with input size 784, two hidden layers of size 64 and one output shape of size 10. 
To achieve reproducible results, the constructor allows the passing of an optional `seed` parameter, which is an integer that will be used to initialize numpy's random generator. This way the initialization of the thetas can become deterministic.
### Training a network
The networks train method can be used as follows:

    network.train(iterations=1000,
                  alpha=0.001,
                  X_train=X_train,
                  y_train=Y_train,
                  X_val=X_val,
                  y_val=Y_val,
                  show_plots=True,
                  log_intervals={'first' : 1, 'second' : 10},
                  use_feature_scaling=True,
                  use_regularization=True,
                  lambda_regularization=0.01,
                  hidden_layer_shape=[64, 64],                      
                  class_labels=['shirt', 'trousers', ..., 'coat', 'boot'],            
                  pca_threshold=0.99)
`iterations:` Training iterations (float)  
`alpha:` Learning rate (float)  
`X_train:` Training set as 2D numpy array, with each row representing a sample  
`Y_train:` Corresponding training labels as 1D numpy array (not one-hot encoded) e.g. [3, 7, 1, 2, .., 8, 9]  
`X_val:` Validation set (same format as training)  
`Y_val:` Corresponding validation labels (same format as training)  
`show_plots:` Boolean if after training the plots should be shown. (Can be disabled to automate hyperparameter tuning without pyplot blocking after each training run)  
`log_intervals:` Dictionary of values 'first' and 'second' expecting integer values to determine at which intervals validation data should be calculated and logged.   


The 'first' value indicates after how many iterations the following metrics are logged:

 1. Train Accuracy
 2. Validation Accuracy
 3. Train Loss
 4. Validation Loss

The 'second' value indicates after how many iterations the following metrics are logged:

 1. Per-Class Precision
 2. Per-Class Recall
 3. Per-Class F1 Score
 4. Confusion Matrix
 
 Since these calculations are quite computationally heavy when done frequently, we recommend to set this value to > 10 to achieve faster training. 
 
`lambda_regularization:` L1 regularization parameter  
`class_labels:` A string array of descriptions for the class labels e.g. ["shirt", "trousers", ...]   **Important:** Train will not work if the length of class_labels differs from the amount of classes apparent in the training / validation sets.  
`hidden_layer_shape:` Due to an artefact of how we handle loading and saving the network, the shape of the hidden layers has to be additionally passed to the train method, since PCA changes the internal input shape, this is a point for possible future improvement.
`pca_threshold:` The network always uses Principal Component Analysis (PCA), which expects a float value from 0 to 1 that determines how much percent of the variance should be retained.
 
### Evaluation and plotting
Once the training is done, if `show_plots` was set to true, the following plots will be displayed:
1. **Training Loss, Accuracy and Metrics:** Plots the overall training loss, accuracy, as well as the precision, 
recall and f1-score per-class. The former is plotted for every 'first' log interval iteration, whereas the latter 
is plotted for every 'second' log interval iteration.

2. **Confusion Matrix:**
Plots the confusion matrix for in an N x N matrix, where N represents all classes.

3. **Confusion Matrix Live:**
Plots the confusion matrix for every 'second' log interval iterations in an animation to receive a better insight, how dynamic the model is improving.

4. **Absolute Theta Sum per Neuron per Layer:**
Plots the absolute sum of thetas per neuron for every hidden layer to 
give more information about the weights of the networks layers after training.

5. **Principal Component Analysis:**
   1. **Cumulative Explained Variance Ratio:** Indicates the amount of required principal components to fulfill the given threshold of variance for the dataset.
   2. **Original Data: Feature vs. Feature:** Plots some example features of the original dataset against other features. The datapoints are colored according to their class.
   3. **Transformed Data: Principal Component vs. Principal Component:** Plots some example principal components of the transformed dataset against other principal components. The datapoints are colored according to their class.
   4. **Eigenvector vs. Eigenvector:** Plots some example eigenvectors against other eigenvectors


### Saving a network
After a network was trained, it can be saved by calling

    network.save(target_dir="C:/my_checkpoints/")
This will create a file called something like `hyper_params_2023_04-03_18-37-48.npz` in the specified directory.
To load this network, the Network class provides the following method:
### Loading a network and using it for inference
    network_loaded = Network.create_with_hyper_params("hyper_params_2023_04-03_18-37-48.npz")
This network could now be trained further, or used for inference as follows:

    unknown_sample = np.array([0.52,  0.41, ..., 0.52, 0.83])
    result = network_loaded.predict(np.array([unknown_sample]))

Since predict is used to handle all samples at once during training, it expects a list of samples as input, so to do inference on a single sample, it has to be passed in a numpy array of size 1 as shown in the example above.
The result will return the values of the final softmaxed layers.
