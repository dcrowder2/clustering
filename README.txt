# clustering
Dakota Crowder
CSCE A415 Machine Learning
Assignment 4

				HOW-TO Run

preprocess.py
Run it and it will prompt you for a file path to the "data" folder, this is the
folder that has the structure \data\(a01-a19)\(p1-p8)\(s01.txt-s60.txt)
which is usually the second folder with extracting the ClusteringData.zip
requires:
python 3
sklearn
scipy
numpy

k-means.py
Run it, and it will prompt you whether you want to use cosine similarity or not
default is N, but in reality if you type anything other than "y" it will think
you mean no and will run with the Eucilean Distance

It will then prompt you with how many clusters do you want,
please only put in 3 or 19, as any other will most likely break the code, and 
there is no reason to do anything other than 3 or 19

It will then run with your selected method and k, printing out the mean
entropy with each selected centroid. It will finish when the centroids are
no longer being changed
