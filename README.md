# MSc-Thesis-final
In this repository, the used codes for my MSc thesis are included. All code in this repository is final and is added after the completion of the thesis.

## How to read the code
Dependencies/packages etc used are found in the dependencies.py files.

To run the used code: use .py-files that start with 9, 10 and 11. They contain fragments from the other files, that were used in the final thesis.
Elaboration of the other files:
1. Contains all preparation for the dataset that is used + the prep for the new data that is used in 11
1b. Contains the code for scatterplots that were used in order to get a first visual and impression of the data.
2. VIF: Performing a variance inflaction factor-analysis on the to-be used regression, which determined that shrinkage regression is to be used for the first analysis.
3. Shrinkage: Holds all code that was used in order to perform CV ridge, lasso and elastic net. Including lambda plots, total table of all models. The best-performing lasso is stored in 9.
4. Secondregression: Combination of all regressions that are used to see difference~price+volume+distri. Also holds major regression that is used to cluster on.
5. Clustering: holds the clustering that is done on the major regression in 4.py. Shows KMeans and KMedoids, with two different ways of compution elbow k. Also holds big table of the final clusters, and PCA plot on the found clusters.
6. a: random forest -diff: contains random forest on the difference on product level. Not used in final thesis, but used to find out what drove difference.
6. b: random forest -marketshare: used in final thesis. Can be found in 10
7. SVM: Holds the different SVM models that are created. Used model can be found in 10.
8. NN: Holds neural network models, does not hold all. Used model can be found in 10.
9. RUN THIS SCRIPT for everything used, except the blackbox (The first lines (read_csv) need to be executed before 10 can be ran)
10. RUN THIS SCRIPT for all black box models
11. RUN THIS SCRIPT for the perfomance of the black box models on the new introduced data
