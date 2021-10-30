# Logistic-Regressor
A binary logistic regressor based on iterative stochastic gradient descent (SGD) with regularization and a comparison on the performance with that of sklearn’s SVM

* Dataset
  * sklearn’s breast cancer data set
  * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
* The iterative solver supports:
  * a 2D grid search with one dimension being the learning rate α and the other dimension being the regularization weight λ
  * mini-batch gradient descent with mini-batch size 8
  * n-fold cross validation with n = 5
  * augmented feature vector [x, 1] so that weights and bias are treated in a uniform way as [w0, w], where w0 replaces the bias
* Compare the run time and performance of the logistic regressor based on gradient descent with that of the SVM based on maximizing margin
