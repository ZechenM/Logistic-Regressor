from ast import get_docstring
from pickle import LIST
from typing import Callable
import numpy as np
from collections import Counter
import copy

from numpy.core.numeric import indices

# Below is a class that implement the binary tree search.
# It shows how recursion is done in a typical binary tree structure.
# You may apply similar ideas (recursion) in CART as you need to recursively split the left and right node until a stop criterion is met.
# For your reference only.
NULL = 0


class bsearch(object):
    '''
    binary search tree, with public functions of search, insert and traversal
    '''

    def __init__(self, value):
        self.value = value
        self.left = self.right = NULL

    def search(self, value):
        if self.value == value:
            return True
        elif self.value > value:
            if self.left == NULL:
                return False
            else:
                return self.left.search(value)
        else:
            if self.right == NULL:
                return False
            else:
                return self.right.search(value)

    def insert(self, value):
        if self.value == value:
            return False
        elif self.value > value:
            if self.left == NULL:
                self.left = bsearch(value)
                return True
            else:
                return self.left.insert(value)
        else:
            if self.right == NULL:
                self.right = bsearch(value)
                return True
            else:
                return self.right.insert(value)

    def inorder(self):
        if self.left != NULL:
            self.left.inorder()
        if self != NULL:
            print(self.value, " ", end="")
        if self.right != NULL:
            self.right.inorder()


# -------------------------------Main code starts here-------------------------------------#
class TreeNode(object):
    '''
    A class for storing necessary information at each tree node.
    Every node should be initialized as an object of this class. 
    '''

    def __init__(self, d=None, threshold=None, l_node=None, r_node=None, label=None, is_leaf=False, gini=None, n_samples=None, depth=None):
        '''
        Input:
            d: index (zero-based) of the attribute selected for splitting use. int
                => decided by BestFeature
            threshold: the threshold for attribute d. If the attribute d of a sample is <= threshold, the sample goes to left 
                       branch; o/w right branch. float
                => decided by BestFeature
            l_node: left children node/branch of current node. TreeNode
                => row: all the rows with respect to the left_label
                => col: all the columns
            r_node: right children node/branch of current node. TreeNode
                => row: all the rows with respect to the right_label
                => col: all the columns
            label: the most common label at current node. int/float
                => Collections.mostcommon() ?
            is_leaf: True if this node is a leaf node; o/w False. bool
                => Return true of gi == 0
                => o.w. return false
            gini: stores gini impurity at current node. float
                => returned by BestFeature
            n_samples: number of samples at current node. int
                => X.shape[0]
        '''
        self.d = d
        self.threshold = threshold
        self.l_node = l_node
        self.r_node = r_node
        self.label = label
        self.is_leaf = is_leaf
        self.gini = gini
        self.n_samples = n_samples
        self.depth = depth


def load_data(fdir):
    '''
    Load attribute values and labels from a npy file. 
    Data is assumed to be stored of shape (N, D) where the first D-1 cols are attributes and the last col stores the labels.
    Input:
        fdir: file directory. str
    Output:
        data_x: feature vector. np ndarray
        data_y: label vector. np ndarray
    '''
    data = np.load(fdir)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class CART(object):
    '''
    Classification and Regression Tree (CART). 
    '''

    def __init__(self, max_depth=None):
        '''
        Input:
            max_depth: maximum depth allowed for the tree. int/None.
        Instance Variables:
            self.max_depth: stores the input max_depth. int/inf
            self.tree: stores the root of the tree. TreeNode object
        '''
        self.max_depth = float('inf') if max_depth is None else max_depth
        self.tree = None
        ###############################
        # TODO: your implementation
        # Add anything you need
        ###############################

    def GiniImpurity(self, y):

        # label the samples into 3 different classes
        sample = np.zeros(3)
        # total # of samples
        total = len(y)

        for qs in y:
            sample[qs] += 1

        gi = 1.0

        if len(y) != 0:
            sample = sample / float(total)
            sample = pow(sample, 2)
            gi -= np.sum(sample)

        return gi

    # call GiniImpurity to find out the best split point on one feature
    # output: return threshold, gini impurity of one feature using the best split point
    def BestThreshold(self, feature, y):
        # D: sorted list of unique values from one feature column vector
        D = np.unique(feature)
        # len(threshold) = len(D) - 1
        threshold_list = []
        # corresponding gini impurity for each threshold split
        gi_list = []

        # iterate through D
        for i in range(len(D) - 1):
            threshold = (D[i] + D[i+1]) / 2
            threshold_list.append(threshold)
            # split the feature vector based on the threshold
            left_label = []
            right_label = []
            # j: row index of samples in this attribute column
            for j in range(len(feature)):
                # left child
                if feature[j] <= threshold:
                    left_label.append(y[j])
                # right child
                else:
                    right_label.append(y[j])

            left_weight = len(left_label) / float(len(feature))
            right_weight = len(right_label) / float(len(feature))
            weighted_gi = left_weight * \
                self.GiniImpurity(left_label) + right_weight * \
                self.GiniImpurity(right_label)
            gi_list.append(weighted_gi)

        # find the smallest gini impurity
        if len(gi_list) == 0:
            min_gi = 1e3
            best_threshold = -1
        else:
            min_gi = np.amin(gi_list)
            best_threshold = threshold_list[np.argmin(gi_list)]

        return min_gi, best_threshold

    # call BestThreshold for all features
    # input:
    #   X: partial dataset with respect to the feature <-- threshold of the parent node
    #   y: correponding labels
    # output: the index of the best feature
    def BestFeature(self, X, y):

        # initialize a list to store the Gini Impurity
        feature_gi = []
        feature_threshold = []

        # scan through all the features
        for i in range(X.shape[1]):
            # find out all the gini for all the features := X[:, i] (column vector)
            # y is the same for all the features/iterations
            gi, threshold = self.BestThreshold(X[:, i], y)
            feature_gi.append(gi)
            feature_threshold.append(threshold)

        best_feature_index, best_feature_threshold, best_feature_gini = None, None, None

        # the best feature comes with the smallest gini impurity
        if len(feature_gi):
            best_feature_index = np.argmin(feature_gi)
            best_feature_gini = np.amin(feature_gi)
            best_feature_threshold = feature_threshold[best_feature_index]

        # (d, threshold, gini)
        return best_feature_index, best_feature_threshold, best_feature_gini

    def train(self, X, y):
        '''
        Build the tree from root to all leaves. The implementation follows the pseudocode of CART algorithm.  
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray

        Pay attention to X that is capitalized
        '''
        ###############################
        # TODO: your implementation
        ###############################

        # build the root
        # N: number of samples
        N, D = X.shape
        index, threshold, child_gini = self.BestFeature(X, y)
        label = self.findLabel(y)

        # assume depth(root) = 0
        self.tree = TreeNode()
        self.tree.d = index
        self.tree.threshold = threshold
        self.tree.label = label
        self.tree.gini = self.GiniImpurity(y)
        self.tree.is_leaf = False
        self.tree.n_samples = N
        self.tree.depth = 0

        # preprun the tree at the root
        if self.tree.gini - child_gini < 1e-4:
            self.tree.is_leaf = True
            return

        # BFS
        node_list = [self.tree, X, y]

        while node_list:
            node = node_list.pop(0)
            X = node_list.pop(0)
            y = node_list.pop(0)

            # construct sub-dataset for left and right branch
            mask = X[:, node.d] <= node.threshold
            left_X = X[mask]
            left_y = y[mask]
            right_X = X[~mask]
            right_y = y[~mask]

            node.l_node = TreeNode()
            node.l_node.d, node.l_node.threshold = self.BestFeature(
                left_X, left_y)[:-1]
            node.l_node.gini = self.GiniImpurity(left_y)
            node.l_node.label = self.findLabel(left_y)
            node.l_node.n_samples = left_X.shape[0]
            node.l_node.depth = node.depth + 1

            node.r_node = TreeNode()
            node.r_node.d, node.r_node.threshold = self.BestFeature(
                right_X, right_y)[:-1]
            node.r_node.gini = self.GiniImpurity(right_y)
            node.r_node.label = self.findLabel(right_y)
            node.r_node.n_samples = right_X.shape[0]
            node.r_node.depth = node.depth + 1

            # stopping condition
            # 3. depth = max_depth <=> is_leaf = True

            if node.l_node.depth == self.max_depth:
                node.l_node.is_leaf = True
                node.l_node.l_node = node.l_node.r_node = None
            else:
                node.l_node.is_leaf = False
                node_list.append(node.l_node)
                node_list.append(left_X)
                node_list.append(left_y)

            if node.r_node.depth == self.max_depth:
                node.r_node.is_leaf = True
                node.r_node.l_node = node.r_node.r_node = None
            else:
                node.r_node.is_leaf = False
                node_list.append(node.r_node)
                node_list.append(right_X)
                node_list.append(right_y)

    def buildTree(self, X, y):
        '''
        Build the tree from root to all leaves. The implementation follows the pseudocode of CART algorithm.  
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        Pay attention to X that is capitalized
        '''
        ###############################
        # TODO: your implementation
        ###############################

        # build the root
        # N: number of samples
        N, D = X.shape
        index, threshold, child_gini = self.BestFeature(X, y)
        label = self.findLabel(y)

        # assume depth(root) = 0
        self.tree = TreeNode()
        self.tree.d = index
        self.tree.threshold = threshold
        self.tree.label = label
        self.tree.gini = self.GiniImpurity(y)
        self.tree.is_leaf = False
        self.tree.n_samples = N
        self.tree.depth = 0

        # preprun the tree at the root
        if self.tree.gini - child_gini < 1e-4:
            self.tree.is_leaf = True
            return

        # BFS
        node_list = [self.tree, X, y]

        while node_list:
            node = node_list.pop(0)
            X = node_list.pop(0)
            y = node_list.pop(0)

            # construct sub-dataset for left and right branch
            mask = X[:, node.d] <= node.threshold
            left_X = X[mask]
            left_y = y[mask]
            right_X = X[~mask]
            right_y = y[~mask]

            node.l_node = TreeNode()
            node.l_node.d, node.l_node.threshold = self.BestFeature(
                left_X, left_y)[:-1]
            node.l_node.gini = self.GiniImpurity(left_y)
            node.l_node.label = self.findLabel(left_y)
            node.l_node.n_samples = left_X.shape[0]
            node.l_node.depth = node.depth + 1

            node.r_node = TreeNode()
            node.r_node.d, node.r_node.threshold = self.BestFeature(
                right_X, right_y)[:-1]
            node.r_node.gini = self.GiniImpurity(right_y)
            node.r_node.label = self.findLabel(right_y)
            node.r_node.n_samples = right_X.shape[0]
            node.r_node.depth = node.depth + 1

            # stopping condition
            # 1. 1 class left == gini = 0
            # 2. cannot decrease gini impurity == gini gain < epsilon
            # 3. depth = max_depth <=> is_leaf = True
            # 4. pre-pruning - minimum number of samples in each node
            gini_pure = [node.l_node.gini == 0.0, node.r_node.gini == 0.0]
            gini_gain = [(node.gini - node.l_node.gini) < 0.05,
                         (node.gini - node.r_node.gini) < 0.05]
            depth = [node.l_node.depth == self.max_depth,
                     node.r_node.depth == self.max_depth]
            min_sample = [node.l_node.n_samples < 5, node.r_node.n_samples < 5]

            if gini_pure[0] or gini_gain[0] or depth[0] or min_sample[0] or node.l_node.d is None or node.l_node.threshold is None:
                node.l_node.is_leaf = True
                node.l_node.l_node = node.l_node.r_node = None
            else:
                node.l_node.is_leaf = False
                node_list.append(node.l_node)
                node_list.append(left_X)
                node_list.append(left_y)

            if gini_pure[1] or gini_gain[1] or depth[1] or min_sample[1] or node.r_node.d is None or node.r_node.threshold is None:
                node.r_node.is_leaf = True
                node.r_node.l_node = node.r_node.r_node = None
            else:
                node.r_node.is_leaf = False
                node_list.append(node.r_node)
                node_list.append(right_X)
                node_list.append(right_y)

    def findLabel(self, y):
        if not len(y):
            return -1
        else:
            return Counter(y).most_common(1)[0][0]

    def test(self, X_test):
        '''
        Predict labels of a batch of testing samples. 
        Input:
            X_test: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        N, D = X_test.shape
        pred = []

        # iterate through X_test row-wise
        for i in range(N):
            # node is completely independent from self.tree
            node = self.tree

            while node.is_leaf == False:
                index = node.d
                if X_test[i, index] <= node.threshold:
                    node = node.l_node
                else:
                    node = node.r_node

            pred.append(node.label)

        return np.array(pred)

    def visualize_tree(self):
        '''
        A simple function for tree visualization. 
        Note that this function assumes a class variable called self.tree that stores the root node.
        If your implementation does not satisfy this pre-requisite, this function may generate false visualization.
        You can modify this function to make it fit your implementation.

        In your final submission to gradescope, you should avoid calling this function!
        '''
        print('ROOT: ')

        def print_tree(tree, indent='\t|', dict_tree={}, direct='L'):
            if tree.is_leaf == True:
                dict_tree = {direct: str(tree.label)}
            else:
                print(indent + 'attribute: %d/threshold: %.5f' %
                      (tree.d, tree.threshold))

                if tree.l_node.is_leaf == True:
                    print(indent + 'L -> label: %d' % tree.l_node.label)
                else:
                    print(indent + "L -> ",)
                a = print_tree(tree.l_node, indent=indent + "\t|", direct='L')
                aa = a.copy()

                if tree.r_node.is_leaf == True:
                    print(indent + 'R -> label: %d' % tree.r_node.label)
                else:
                    print(indent + "R -> ",)
                b = print_tree(tree.r_node, indent=indent + "\t|", direct='R')
                bb = b.copy()

                aa.update(bb)
                stri = indent + \
                    'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold)
                if indent != '\t|':
                    dict_tree = {direct: {stri: aa}}
                else:
                    dict_tree = {stri: aa}
            return dict_tree
        try:
            if self.tree is None:
                raise RuntimeError('No tree has been trained!')
        except:
            raise RuntimeError('No self.tree variable!')
        _ = print_tree(self.tree)


def cross_validation_split(X, y, index, fold=5):
    x_split = np.array_split(X, fold)
    y_split = np.array_split(y, fold)

    x_test = x_split.pop(index)
    y_test = y_split.pop(index)

    x_train = np.vstack(x_split)
    y_train = np.hstack(y_split)

    return x_train, y_train, x_test, y_test


def GridSearchCV(X, y, depth=[1, 40]):
    '''
    Grid search and cross validation.
    Try different values of max_depth to observe the performance change. 
    Apply 5-fold cross validation to find the best depth. 
    Input:
        X: full training dataset. Not split yet. np ndarray
        y: full training labels. Not split yet. np ndarray
        depth: [minimum depth to consider, maximum depth to consider]. list of integers
    Output:
        best_depth: the best max_depth value from grid search results. int
        best_acc: the validation accuracy corresponding to the best_depth. float
        best_tree: a decision tree object that is trained with 
                   full training dataset and best max_depth from grid search. instance
    '''
    ###############################
    # TODO: your implementation
    ###############################

    depth_list = []
    for depth in range(depth[0], depth[1]+1):
        depth_list.append(depth)

    acc_list = []
    tree_list = []

    # iterate through depth
    for depth in depth_list:
        # 5-fold cross validation
        acc_cv = np.ones(5)
        for index in range(5):
            x_train, y_train, x_valid, y_valid = cross_validation_split(
                X, y, index)

            # CART instance
            tree = CART(depth)
            tree.buildTree(x_train, y_train)

            # accuracy counter
            cnt = 0
            # apply CART tree built from x_train and y_train on x_valid and y_valid
            for pred_label, gt_label in zip(tree.test(x_valid), y_valid):
                if pred_label == gt_label:
                    cnt += 1

            # accuracy = # of correctly classified samples / # of samples
            acc_cv[index] = cnt / len(y_valid)

        # average validation accuracy => want to maximize accuracy
        acc_list.append(np.mean(acc_cv))
        tree_list.append(tree)

    # best_index == the index of largest element in acc_list == depth_list[i] == best_depth - 1
    # the best max_depth value from grid search results
    best_depth = np.argmax(acc_list) + 1
    best_acc = np.amax(acc_list)

    # train the decision tree with full training dataset and best_depth
    best_tree = CART(best_depth)
    best_tree.buildTree(X, y)

    return best_depth, best_acc, best_tree


x = np.array([[6,9],
              [3,6],
              [2,4],
              [7,2]])
y = np.array([1,0,1,0])
tree = CART(1)
tree.train(x,y)
tree.visualize_tree()
# main
# NOTE: Do not change anything below
X_train, y_train = load_data('winequality-red-train.npy')
best_depth, best_acc, best_tree = GridSearchCV(X_train, y_train, [1, 40])
print('Best depth from 5-fold cross validation: %d' % best_depth)
print('Best validation accuracy: %.5f' % (best_acc))
