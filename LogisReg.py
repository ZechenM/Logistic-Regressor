import time
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import numpy.linalg as npla
from random import randrange


class Binary_Classifier(object):

    def __init__(self, train_data, train_target):
        """
        Data is pre-loaded and passed to __init__ directly. train_data is the train feature matrix
        and train_target is the train label vector. You can store both
        training features and target vector as instance variables
        """

        self.row, self.col = train_data.shape

        # augumented feature vector
        self.x = np.ones((self.row, self.col+1))
        # self.x[:, :-1] = train_data
        self.x[:, 1:] = train_data

        self.y = np.array(train_target)

        # validation error
        self.e = 1e6

    def cross_validation_split(self, index, fold=5):
        # split the datatset into 5 bins
        # x/y_split is a list of 5 arrays

        # x_copy = self.x

        # # standaradization
        # for i in range(1, x_copy.shape[1]):
        #     tmp = x_copy[:, i]
        #     x_copy[:, i] = (tmp - np.mean(tmp)) / np.std(tmp)

        x_split = np.array_split(self.x, fold)
        y_split = np.array_split(self.y, fold)

        # index = index of validation set in x/y_split
        x_test = x_split.pop(index)

        # np.vstack will row-wise concatenate the arrays in a list to one array
        x_train = np.vstack(x_split)

        y_test = y_split.pop(index)
        # since y is 1-d array, have to concatenate them column-wise
        y_train = np.hstack(y_split)

        return x_train, x_test, y_train, y_test

    def logistic_training(self, alpha, lam, nepoch, epsilon):
        """
        Training process of logistic regression.
        alpha: learning rate
        lam: regularization strength
        nepoch: number of epochs to train.
        eposilon: early stop condition.

        The use of these parameters is the same as that in program #1.
        You implementation must include 5-fold cross validation,
        Hint: You can store the weight as an instance variable,
        so other functions can directly use them
        """
        # find the best w and b until error < epsilon

        # 2-D grid search - outer: alpha; inner: lam
        # lr: learning rate
        # rw: regularization weight
        lr = alpha[0]
        while lr <= alpha[1]:
            rw = lam[0]
            while rw <= lam[1]:
                e = np.ones(5) * 1e3

                w = np.zeros(self.x.shape[1])

                # perform 5-fold cross validation
                for index in range(5):
                    x_train, x_test, y_train, y_test = self.cross_validation_split(
                        index)

                    w = np.random.uniform(0, 1, self.x.shape[1]) / 100
                    w[0] = 0
                    # w = np.zeros(self.x.shape[1])

                    for i in range(nepoch):
                        # define the size of the mini-batch
                        batch = 8
                        # count the number of mini-batches needed
                        # if n = 5, batch = 2, size = 2
                        # then first two bins: one w/ size = 3, another w/ size = 2
                        size = int(len(x_train) / batch)

                        # split x_train and y_train
                        x_split = np.array_split(x_train, size)

                        # x_train <=> y_train
                        y_split = np.array_split(y_train, size)

                        # sigmoid_list = []

                        for x_sub, y_sub in zip(x_split, y_split):
                            z = x_sub.dot(w)

                            sigmoid = 1/(1 + np.exp(-z))
                            # sigmoid_list.append(sigmoid)

                            w = w + lr * \
                                (x_sub.T).dot(y_sub - sigmoid) - rw*w

                            # if w became too big, discard the current value and normalize it back to default
                            if npla.norm(w) > 1e3:
                                w = np.random.uniform(
                                    0, 1, self.x.shape[1]) / 100
                                w[0] = 0
                                break

                    # after all the epochs, calculate the cross entropy
                    # use the trained model to make predictions on x_test
                    z = x_test.dot(w)
                    # y_pred = sigmoid(z)
                    y_pred = 1 / (1 + np.exp(-z))
                    # update the validation error
                    e[index] = np.sum(-(y_test * np.log(y_pred) +
                                        (1 - y_test) * np.log(1 - y_pred)))

                # average validation error
                e_avg = np.mean(e)

                print("E_AVG: ", e_avg)

                # store the best alpha and lam
                if e_avg < self.e:
                    print("Good E_AVG: ", e_avg)
                    self.e = e_avg
                    self.rw = rw
                    self.lr = lr
                    self.w = w

                # increment regularization weight lambda
                rw *= 10

            # increment learning rate alpha
            lr *= 10

        print("Learning rate: ", self.lr)
        print("Regularization weight: ", self.rw)

        w = np.random.uniform(0, 1, self.x.shape[1]) / 100
        w[0] = 0
        for i in range(nepoch):

            # define the size of the mini-batch
            batch = 8
            # count the number of mini-batches needed
            # if n = 5, batch = 2, size = 2
            # then first two bins: one w/ size = 3, another w/ size = 2
            size = int(len(self.x) / batch)

            # split self.x and self.y
            x_split = np.array_split(self.x, size)

            # self.x <=> self.y
            y_split = np.array_split(self.y, size)

            for x_sub, y_sub in zip(x_split, y_split):
                z = x_sub.dot(w)

                sigmoid = 1/(1 + np.exp(-z))
                # sigmoid_list.append(sigmoid)

                w = w + self.lr * \
                    (x_sub.T).dot(y_sub - sigmoid) - self.rw*w

        self.w = w
        print("w is ", self.w)

    def logistic_testing(self, testX):
        # augumented feature
        """
        Use your trained weight and bias to compute the predicted y values,
        Predicted y values should be 0 or 1. return the numpy array in shape n*1
        """
        # dimensions of testX
        m, n = testX.shape

        # augumented feature vector
        testX_copy = np.ones((m, n+1))
        # everything except the last column of testX_copy = testX
        testX_copy[:, 1:] = testX

        z = testX_copy.dot(self.w)
        sigmoid = 1 / (1 + np.exp(-z))

        for i in range(len(sigmoid)):
            if sigmoid[i] > 0.5:
                sigmoid[i] = 1
            else:
                sigmoid[i] = 0

        return sigmoid.reshape(m, 1)

    def svm_training(self, gamma, C):
        """
        Training process of the support vector machine.
        gamma, C: grid search parameters

        As softmargin SVM can handle nonlinear boundaries and outliers much better than logistic regression,
        we do not perform 3-fold validation here (just one training run with 90-10 training-validation split).
        Furthmore, you are allowed to use SVM's built-in grid search method.

        This function will be a "wrapper" around sklearn.svm.SVC with all other parameters take the default values.
        Please consult sklearn.svm.SVC documents to see how to use its "fit" and "predict" functions.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """

        # set up parameters for GridSearchCV
        p_grid = {"C": C, "gamma": gamma}
        # estimator: support vector classifier
        svm = SVC()
        # GridSearch CV
        self.clf = GridSearchCV(svm, p_grid)
        # train the model
        self.clf.fit(self.x, self.y)

    def svm_testing(self, testX):
        """
        Use your trained SVM to return the numpy array in shape n*1, predicted y values should be 0 or 1
        """

        # dimensions of testX
        m, n = testX.shape

        # augumented feature vector
        testX_copy = np.ones((m, n+1))
        # everything except the last column of testX_copy = testX
        testX_copy[:, 1:] = testX

        y = self.clf.predict(testX_copy)
        # print("SVM Result: ", y)

        return y.reshape(m, 1)


# Dataset preparation: Dataset is divided into 90% and 10%
# 90% for you to perform n-fold cross validation and 10% for autograder to validate your performance.
################## PLEASE DO NOT MODIFY ANYTHING! ##################
dataset = load_breast_cancer(as_frame=True)
train_data = dataset['data'].sample(
    frac=0.9, random_state=0)  # random state is a seed value
train_target = dataset['target'].sample(
    frac=0.9, random_state=0)  # random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)


# Model training: You are allowed to change the last two inputs for model.logistic_training
################## PLEASE DO NOT MODIFY ANYTHING ELSE! ##################
model = Binary_Classifier(train_data, train_target)

# Logistic Regression
logistic_start = time.time()
# I changed 300 to 30
model.logistic_training([10**-10, 10], [10e-10, 1e10], 300, 0)
logistic_end = time.time()

# SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()
