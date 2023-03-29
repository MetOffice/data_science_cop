#!/usr/bin/env python	
#
# Cyril J Morcrette (2020), Met Office, UK
#
# Within UKMO use "module load scitools/experimental-current" at the command line before running this.
#
# Import some modules (possibly more than are strictly needed, but I did not remove what was not needed).

import numpy as np

import matplotlib.pyplot as plt
import iris
# If external to UKMO, instructions for using Iris are here: https://scitools.org.uk/iris/docs/v2.0/installing.html
from netCDF4 import Dataset

import tensorflow as tf
import os
from tensorflow import keras 
from tensorflow.keras import layers
import pathlib
import warnings

# cjm_functions can be found in the same repository as this file.
from cjm_functions import load_in_one_file_for_cbh_ml

# Start of functions

# End of functions

def main():
    warnings.filterwarnings("ignore",".*HybridHeightFactory*.")
    # Define the deep neural network
    n_nodes=256
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_nodes, input_dim=210, activation='relu')) # This defines the input layer AND the first hidden layer
    model.add(keras.layers.Dense(n_nodes, activation='relu'))                # This is a hidden layer
    model.add(keras.layers.Dense(n_nodes, activation='relu'))                # This is a hidden layer
    model.add(keras.layers.Dense(n_nodes, activation='relu'))                # This is a hidden layer
    model.add(keras.layers.Dense(n_nodes, activation='relu'))                # This is a hidden layer
    model.add(keras.layers.Dense(     70, activation='softmax'))             # This is the output layer 
                                                                # A softmax is used in output layer as we want the final 
                                                                # output to be intepretable as a probability.
    opt = keras.optimizers.Adam(learning_rate=1.0e-4)
    # In above lines, you can: add more layers
    #                          change number of nodes
    #                          change activation functions
    #                          add drop-out (randomly ignore a fraction of the nodes while training)
    #                          add regularization (e.g. max norms)
    #                          change the optimzer (opt)
    #                          change the learning rate
    model.summary()
    # Writes out a summary of the network architecture
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', "categorical_crossentropy"])
    # Try playing with loss= changing how the loss is quantified.    
    #
    # Write things out to file (for reading back in later to make predictions).
    model_json=model.to_json()
    with open("cbh_ml_model.json", "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    #
    # Define directory where training data is stored.
    directory_str= os.environ["SCRATCH"] + "/cbh_data/train_individual_files/"
    directory = os.fsencode(directory_str)
    #
    # Set up arrays to store the accuracy as one loops through multiple sets of multiples epochs.
    keep_acc_train=np.empty([1,1])+np.nan
    keep_acc_val  =np.empty([1,1])+np.nan
    # The variable called super_loop counts how many times we loop through reading in the entire data set (one file at a time).
    # Note also that there is "epoch=" in the model.fit line. 
    # One could argue that should be set to 1, because we are only reading and training on 
    # a small part of the entire data and we do not want to overfit to that data by doing multiple epochs there.
    # i.e. set epoch=1 and n_super_loop=20
    # Alternatively, having taken the time to read small portion of the data, perhaps we should do 
    # a few passes through it e.g. epoch=20 but then only read through the entire data set once n_super_loop=1 
    # Basically, feel free to explore.
    n_super_loop=5
    # You might want to try changing the number of super_loops.
    for super_loop in np.arange(0,n_super_loop):
        file_no=0
        for file in os.listdir(directory):
            file_no=file_no+1
            filename = directory_str+os.fsdecode(file)
            print('Super_Loop=',super_loop,'Loading file_no',file_no,filename)
            result=load_in_one_file_for_cbh_ml(filename)
            big_data=np.transpose(result['data'])
            #
            # Manually shuffle data (you can do this as part of the model.fit call, 
            # but doing it manually means you can plot the data at this point and visualise what is going on).
            shuffled_big_data=np.random.permutation(big_data)
            # Must shuffle array and *then* take X and y from it 
            # (rather than take X and y and then shuffle) to make sure things line up.
            # X is the long vector of inputs (temp, qv, pres)
            # y is the one-hot encoded vector of where cloud base is.
            X=shuffled_big_data[:,  0:210]
            y=shuffled_big_data[:,210:280]
            n_samples,tmp=X.shape
            # Use 80% of the data for training and the rest for validation.
            n_train=int(0.8*n_samples)
            trainX, testX = X[:n_train, :], X[n_train:, :]
            trainy, testy = y[:n_train], y[n_train:]
            # You might want to see the impact of changing the batch_size
            history=model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1, batch_size=1000)
            keep_acc_train=np.append(keep_acc_train,history.history['accuracy'])
            keep_acc_val  =np.append(keep_acc_val,  history.history['val_accuracy'])
            # Write out some metrics to track progress. You need to read these back in later and plot them.
            fileout='cbh_ml_timeseries_acc_train.txt'
            np.savetxt(fileout, np.ones((1,1))*keep_acc_train, fmt='%10.7f')
            fileout='cbh_ml_timeseries_acc_val.txt'
            np.savetxt(fileout, np.ones((1,1))*keep_acc_val, fmt='%10.7f')
        # Write out the weights once per super_loop
        model.save_weights("cbh_ml_latest_saved_weights.h5")

    print('All training completed successfully!')

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()




