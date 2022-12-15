from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode
from collections import Counter

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
    K = 6
    
    N = train_image_feats.shape[0]
    M = test_image_feats.shape[0]
    d = train_image_feats.shape[1]
    
    # calculate the distance between each test image and training images
    dists = distance.cdist(test_image_feats, train_image_feats, metric='cityblock')
    test_predicts = []
    # iterate through distances of each test image
    for dist in dists:
        # find the indices that would sort an array.
        idx = np.argsort(dist)
        # find K labels which are closest to the test image
        label = [train_labels[idx[i]] for i in range(K)]
        # the label occurs most times
        label_final = Counter(label).most_common(1)[0][0]
        # then that is the predicted label of test sets
        test_predicts.append(label_final)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
