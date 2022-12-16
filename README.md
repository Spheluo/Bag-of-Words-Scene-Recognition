# Bag-of-Words-Scene-Recognition
## implementation of Bag of Words (BoW) Scene Recognition

### you can get the training and testing set from [here](https://drive.google.com/file/d/1ycutiD0rsnnefWlEgs0u0KmW2uXXXTKE/view?usp=share_link).

### 1. Get feature vectors from 'tiny images' and use KNN classifier to predict the label of test data where the accuracy is about 0.23. The command is as follows:

python3 p1.py --feature 'tiny_image' --classifier 'nearest_neighbor' 

### 2. Get feature vectors from 'Bag of Sift', and use KNN classifier to predict the label of test data where the accuracy is about 0.61. The command is as follows:

python3 p1.py --feature 'bag_of_sift' --classifier 'nearest_neighbor' 
