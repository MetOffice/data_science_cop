# understanding-clouds-kaggle
This repository hosts the Data Science Community of Practice Understanding Clouds challenge. This challenge was built 
upon the [Kaggle Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization/) 
challenge hosted by the Max Planck Institute for Meteorology. 

The original challenge required users to segment regions belonging to each of four classes - Fish, Flower, Gravel, and 
Sugar. However, this challenge was adapted to a image classification task for the Data Science Community of Practice.
A set of 224x224 images that consist of a single class have been extracted, and randomly separated into training and 
test sets. These will be used to train image classification algorithms that can be evaluated against the test images
using a provided script. 

This repository contains the following code:

__src/produce_test_train__

Scripts used to produce train / test image sets and labels

__src/example_code__

Examples of image classification algorithms that may be used to develop classifiers

__src/evaluate__

Script for evaluating classification of test image set

__TODO:__ 
- Produce example classification script
- Produce scoring script
- Create and set universal environment
- Add links to sharepoint, kick off meeting, monthly catchups, etc.