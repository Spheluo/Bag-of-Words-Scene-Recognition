from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
from tqdm import tqdm

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    
    image_feats = []
    
    start_time = time()
    print("Construct bags of sifts...")
    
    for path in tqdm(image_paths):
        img = np.asarray(Image.open(path),dtype='float32')
        _, descriptor = dsift(img, step=[3,3], fast=True)
        # calculate the distance between vocab and features of training set
        # vocab.shape = (vocab_size, 128), descriptor.shape = (keypoint, 128)
        # dist.shape = (vocab_size, keypoints)
        dist = distance.cdist(vocab, descriptor, metric='cityblock')
        # find the index of a vocabulary closest to each feature of samples from n=150 vocabulary
        idx = np.argmin(dist, axis=0)
        # calculate the appearence time of each vocabulary
        hist, _ = np.histogram(idx, bins=len(vocab))
        # normalize histogram
        hist_norm = [float(i)/sum(hist) for i in hist]
        image_feats.append(hist_norm)
    # turn feature into 2-D numpy array
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print(f"It takes {((end_time - start_time)/60):.2f} minutes to construct bags of sifts.")
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
